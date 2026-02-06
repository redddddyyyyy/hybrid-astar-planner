"""
Dynamic path planner with real-time replanning.

Handles dynamic obstacles by monitoring path validity and
triggering replanning when obstacles invalidate current path.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np

from .hybrid_astar import HybridAStar, PlannerConfig, smooth_path
from .vehicle import Vehicle, State
from .grid import OccupancyGrid, GridConfig


@dataclass
class DynamicObstacle:
    """Represents a dynamic obstacle with position and velocity."""

    x: float
    y: float
    radius: float
    vx: float = 0.0  # Velocity in x
    vy: float = 0.0  # Velocity in y
    id: int = 0

    def predict_position(self, dt: float) -> Tuple[float, float]:
        """Predict position after dt seconds."""
        return (self.x + self.vx * dt, self.y + self.vy * dt)

    def update(self, dt: float) -> None:
        """Update position based on velocity."""
        self.x += self.vx * dt
        self.y += self.vy * dt


@dataclass
class ReplanningConfig:
    """Configuration for replanning behavior."""

    check_interval: float = 0.1  # Path validity check interval (s)
    lookahead_time: float = 2.0  # Time to predict obstacle positions (s)
    safety_margin: float = 1.0  # Additional safety margin around obstacles (m)
    min_replan_interval: float = 0.5  # Minimum time between replans (s)
    path_check_resolution: float = 0.5  # Resolution for path validity checks (m)


class DynamicPlanner:
    """
    Path planner with dynamic obstacle handling and replanning.

    Continuously monitors path validity against dynamic obstacles
    and triggers replanning when necessary.
    """

    def __init__(
        self,
        base_grid: OccupancyGrid,
        vehicle: Vehicle | None = None,
        planner_config: PlannerConfig | None = None,
        replan_config: ReplanningConfig | None = None
    ):
        self.base_grid = base_grid
        self.vehicle = vehicle or Vehicle()
        self.planner_config = planner_config or PlannerConfig()
        self.replan_config = replan_config or ReplanningConfig()

        # State
        self.current_path: List[State] | None = None
        self.current_position_idx: int = 0
        self.goal: State | None = None
        self.dynamic_obstacles: List[DynamicObstacle] = []

        # Callbacks
        self.on_path_updated: Callable[[List[State]], None] | None = None
        self.on_replan_triggered: Callable[[str], None] | None = None

        # Threading
        self._running = False
        self._monitor_thread: threading.Thread | None = None
        self._last_replan_time = 0.0
        self._lock = threading.Lock()

    def set_goal(self, goal: State) -> None:
        """Set planning goal."""
        self.goal = goal

    def update_position(self, position: State) -> None:
        """Update current vehicle position along path."""
        if self.current_path is None:
            return

        # Find closest point on path
        min_dist = float('inf')
        min_idx = self.current_position_idx

        for i in range(self.current_position_idx, len(self.current_path)):
            dist = position.distance_to(self.current_path[i])
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        self.current_position_idx = min_idx

    def add_obstacle(self, obstacle: DynamicObstacle) -> None:
        """Add a dynamic obstacle."""
        with self._lock:
            self.dynamic_obstacles.append(obstacle)

    def remove_obstacle(self, obstacle_id: int) -> None:
        """Remove a dynamic obstacle by ID."""
        with self._lock:
            self.dynamic_obstacles = [
                o for o in self.dynamic_obstacles if o.id != obstacle_id
            ]

    def update_obstacle(self, obstacle_id: int, x: float, y: float,
                       vx: float = None, vy: float = None) -> None:
        """Update obstacle position and optionally velocity."""
        with self._lock:
            for obs in self.dynamic_obstacles:
                if obs.id == obstacle_id:
                    obs.x = x
                    obs.y = y
                    if vx is not None:
                        obs.vx = vx
                    if vy is not None:
                        obs.vy = vy
                    break

    def update_obstacles(self, dt: float) -> None:
        """Update all obstacle positions based on velocities."""
        with self._lock:
            for obs in self.dynamic_obstacles:
                obs.update(dt)

    def plan_initial(self, start: State, goal: State) -> List[State] | None:
        """Plan initial path from start to goal."""
        self.goal = goal

        # Create grid with current obstacles
        grid = self._create_planning_grid()

        # Plan
        planner = HybridAStar(grid, self.vehicle, self.planner_config)
        path = planner.plan(start, goal)

        if path is not None:
            path = smooth_path(path)
            with self._lock:
                self.current_path = path
                self.current_position_idx = 0

            if self.on_path_updated:
                self.on_path_updated(path)

        return path

    def replan_from_current(self) -> List[State] | None:
        """Replan from current position to goal."""
        if self.current_path is None or self.goal is None:
            return None

        # Current position
        if self.current_position_idx >= len(self.current_path):
            return self.current_path  # Already at goal

        current = self.current_path[self.current_position_idx]

        # Replan
        return self.plan_initial(current, self.goal)

    def check_path_validity(self) -> Tuple[bool, str]:
        """
        Check if current path is still valid.

        Returns:
            (is_valid, reason) - True if path is valid, reason if not
        """
        if self.current_path is None:
            return True, ""

        with self._lock:
            obstacles = list(self.dynamic_obstacles)

        # Check remaining path
        remaining_path = self.current_path[self.current_position_idx:]

        for i, state in enumerate(remaining_path):
            # Estimate time to reach this point
            dist_along_path = i * self.replan_config.path_check_resolution
            time_to_point = dist_along_path / 2.0  # Assume 2 m/s

            # Check against predicted obstacle positions
            for obs in obstacles:
                # Predict obstacle position
                pred_x, pred_y = obs.predict_position(time_to_point)

                # Distance to path point
                dist = np.sqrt((state.x - pred_x)**2 + (state.y - pred_y)**2)

                # Check collision with safety margin
                threshold = obs.radius + self.vehicle.config.width / 2 + self.replan_config.safety_margin

                if dist < threshold:
                    return False, f"Obstacle {obs.id} blocks path at ({state.x:.1f}, {state.y:.1f})"

        return True, ""

    def start_monitoring(self) -> None:
        """Start background thread for path monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop background monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None

    def _monitor_loop(self) -> None:
        """Background loop for path validity monitoring."""
        while self._running:
            time.sleep(self.replan_config.check_interval)

            if self.current_path is None:
                continue

            # Check path validity
            is_valid, reason = self.check_path_validity()

            if not is_valid:
                # Check minimum replan interval
                now = time.time()
                if now - self._last_replan_time < self.replan_config.min_replan_interval:
                    continue

                # Trigger replan
                if self.on_replan_triggered:
                    self.on_replan_triggered(reason)

                new_path = self.replan_from_current()
                self._last_replan_time = now

                if new_path is None:
                    # Could not find valid path
                    pass

    def _create_planning_grid(self) -> OccupancyGrid:
        """Create planning grid with current obstacles."""
        # Copy base grid
        grid = OccupancyGrid(self.base_grid.config)
        grid.grid = self.base_grid.grid.copy()

        # Add dynamic obstacles with safety margin
        with self._lock:
            for obs in self.dynamic_obstacles:
                expanded_radius = obs.radius + self.replan_config.safety_margin
                grid.add_circle(obs.x, obs.y, expanded_radius)

        return grid

    def get_remaining_path(self) -> List[State]:
        """Get remaining path from current position."""
        if self.current_path is None:
            return []
        return self.current_path[self.current_position_idx:]


class ObstacleTracker:
    """
    Tracks and predicts dynamic obstacle movements.

    Uses simple linear prediction with velocity estimation.
    """

    def __init__(self, history_length: int = 10):
        self.history_length = history_length
        self.obstacle_history: dict[int, List[Tuple[float, float, float]]] = {}

    def update(self, obstacle_id: int, x: float, y: float, timestamp: float) -> None:
        """Add new observation for an obstacle."""
        if obstacle_id not in self.obstacle_history:
            self.obstacle_history[obstacle_id] = []

        history = self.obstacle_history[obstacle_id]
        history.append((x, y, timestamp))

        # Keep only recent history
        if len(history) > self.history_length:
            history.pop(0)

    def estimate_velocity(self, obstacle_id: int) -> Tuple[float, float]:
        """Estimate obstacle velocity from history."""
        if obstacle_id not in self.obstacle_history:
            return (0.0, 0.0)

        history = self.obstacle_history[obstacle_id]
        if len(history) < 2:
            return (0.0, 0.0)

        # Use last two observations
        x1, y1, t1 = history[-2]
        x2, y2, t2 = history[-1]

        dt = t2 - t1
        if dt < 1e-6:
            return (0.0, 0.0)

        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt

        return (vx, vy)

    def predict_position(self, obstacle_id: int, dt: float) -> Tuple[float, float] | None:
        """Predict obstacle position after dt seconds."""
        if obstacle_id not in self.obstacle_history:
            return None

        history = self.obstacle_history[obstacle_id]
        if not history:
            return None

        x, y, _ = history[-1]
        vx, vy = self.estimate_velocity(obstacle_id)

        return (x + vx * dt, y + vy * dt)
