"""Hybrid A* path planning algorithm."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .vehicle import Vehicle, VehicleConfig, State
from .grid import OccupancyGrid
from .dubins import compute_dubins_path


@dataclass
class PlannerConfig:
    """Hybrid A* planner configuration."""

    # Grid discretization for state space
    xy_resolution: float = 1.0  # Position discretization (m)
    theta_resolution: float = math.radians(5)  # Heading discretization (rad)

    # Motion primitives
    num_steer_angles: int = 5  # Number of steering angle samples
    step_size: float = 1.5  # Motion primitive length (m)
    include_reverse: bool = True  # Allow reverse motions

    # Cost weights
    steer_cost: float = 1.0  # Penalty for steering
    steer_change_cost: float = 2.0  # Penalty for changing steering direction
    reverse_cost: float = 3.0  # Penalty for reverse motion
    direction_change_cost: float = 5.0  # Penalty for switching direction

    # Safety
    safety_margin: float = 0.5  # Collision check inflation (m)

    # Search limits
    max_iterations: int = 50000  # Maximum search iterations

    # Goal tolerance
    goal_xy_tolerance: float = 1.0  # Position tolerance (m)
    goal_theta_tolerance: float = math.radians(10)  # Heading tolerance (rad)


@dataclass(order=True)
class Node:
    """Search node for A* algorithm."""

    f_cost: float  # Total cost (g + h)
    state: State = field(compare=False)
    g_cost: float = field(compare=False)  # Cost from start
    parent: Node | None = field(compare=False, default=None)
    steer: float = field(compare=False, default=0.0)  # Steering to reach this node
    reverse: bool = field(compare=False, default=False)  # Reverse to reach this node

    def get_index(self, config: PlannerConfig) -> Tuple[int, int, int]:
        """Get discretized state index for closed set."""
        ix = int(self.state.x / config.xy_resolution)
        iy = int(self.state.y / config.xy_resolution)
        itheta = int(self.state.theta / config.theta_resolution) % int(
            2 * math.pi / config.theta_resolution
        )
        return (ix, iy, itheta)


class HybridAStar:
    """
    Hybrid A* path planner for non-holonomic vehicles.

    Combines grid-based A* search with continuous state tracking
    to find kinematically feasible paths.
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        vehicle: Vehicle | None = None,
        config: PlannerConfig | None = None
    ):
        self.grid = grid
        self.vehicle = vehicle or Vehicle()
        self.config = config or PlannerConfig()

        # Motion primitives
        self.primitives = self.vehicle.get_motion_primitives(
            num_angles=self.config.num_steer_angles,
            step_size=self.config.step_size,
            include_reverse=self.config.include_reverse
        )

    def plan(self, start: State, goal: State) -> List[State] | None:
        """
        Plan a path from start to goal.

        Args:
            start: Start state (x, y, theta)
            goal: Goal state (x, y, theta)

        Returns:
            List of states forming the path, or None if no path found
        """
        # Validate start and goal
        if self._check_collision(start):
            print("Start state is in collision!")
            return None

        if self._check_collision(goal):
            print("Goal state is in collision!")
            return None

        # Initialize search
        start_node = Node(
            f_cost=self._heuristic(start, goal),
            state=start,
            g_cost=0.0
        )

        open_set: List[Node] = [start_node]
        closed_set: Dict[Tuple[int, int, int], Node] = {}
        open_set_tracker: Dict[Tuple[int, int, int], float] = {}
        iterations = 0

        while open_set and iterations < self.config.max_iterations:
            iterations += 1

            # Pop node with lowest f_cost
            current = heapq.heappop(open_set)
            current_index = current.get_index(self.config)

            # Skip if already processed
            if current_index in closed_set:
                continue

            closed_set[current_index] = current

            # Check if goal reached
            if self._is_goal(current.state, goal):
                print(f"Path found in {iterations} iterations!")
                return self._reconstruct_path(current)

            # Try Dubins shortcut every 10 iterations
            # WHY every 10? Computing a Dubins curve + collision checking it
            # costs time. Doing it EVERY iteration would slow us down. Every
            # 10 is a good balance — frequent enough to catch shortcuts early,
            # rare enough to not waste time when we're far from the goal.
            if iterations % 10 == 0:
                shortcut = self._try_dubins_shortcut(current, goal)
                if shortcut is not None:
                    print(f"Dubins shortcut found at iteration {iterations}!")
                    return shortcut

            # Expand neighbors using motion primitives
            for steer, distance, reverse in self.primitives:
                # Compute new state
                new_state = self.vehicle.step(
                    current.state, steer, distance, reverse
                )

                # Check bounds
                if not self.grid.in_bounds(new_state.x, new_state.y):
                    continue

                # Check collision
                if self._check_collision(new_state):
                    continue

                # Check path collision (intermediate states)
                if self._check_path_collision(current.state, new_state, steer, reverse):
                    continue

                # Compute costs
                g_cost = current.g_cost + self._compute_cost(
                    current, steer, distance, reverse
                )
                h_cost = self._heuristic(new_state, goal)
                f_cost = g_cost + h_cost

                new_node = Node(
                    f_cost=f_cost,
                    state=new_state,
                    g_cost=g_cost,
                    parent=current,
                    steer=steer,
                    reverse=reverse
                )

                # Check if already in closed set
                new_index = new_node.get_index(self.config)
                if new_index in closed_set:
                    continue

                # Only push if we found a better path to this cell
                if new_index in open_set_tracker and g_cost >= open_set_tracker[new_index]:
                    continue

                open_set_tracker[new_index] = g_cost
                heapq.heappush(open_set, new_node)

        print(f"No path found after {iterations} iterations")
        return None

    def _heuristic(self, state: State, goal: State) -> float:
        """
        Compute heuristic cost to goal.

        Uses non-holonomic heuristic combining Euclidean distance
        and Reeds-Shepp path length estimate.
        """
        # Euclidean distance
        dx = goal.x - state.x
        dy = goal.y - state.y
        dist = math.sqrt(dx * dx + dy * dy)

        # Heading difference penalty
        dtheta = abs(self._normalize_angle(goal.theta - state.theta))
        heading_cost = dtheta * self.vehicle.config.min_turn_radius

        return dist + 0.5 * heading_cost

    def _compute_cost(
        self,
        parent: Node,
        steer: float,
        distance: float,
        reverse: bool
    ) -> float:
        """Compute edge cost."""
        cost = distance

        # Steering cost
        cost += self.config.steer_cost * abs(steer)

        # Steering change cost
        if parent.parent is not None:
            cost += self.config.steer_change_cost * abs(steer - parent.steer)

        # Reverse cost
        if reverse:
            cost += self.config.reverse_cost * distance

        # Direction change cost
        if parent.parent is not None and reverse != parent.reverse:
            cost += self.config.direction_change_cost

        return cost

    def _check_collision(self, state: State) -> bool:
        """Check if vehicle at state collides with obstacles."""
        footprint = self.vehicle.get_footprint(state)
        return self.grid.check_collision_polygon(footprint)

    def _check_path_collision(
        self,
        start: State,
        end: State,
        steer: float,
        reverse: bool
    ) -> bool:
        """Check collision along path between two states."""
        # Sample intermediate states
        num_checks = 3
        for i in range(1, num_checks):
            t = i / num_checks
            intermediate = self.vehicle.step(
                start, steer, t * self.config.step_size, reverse
            )
            if self._check_collision(intermediate):
                return True
        return False

    def _is_goal(self, state: State, goal: State) -> bool:
        """Check if state is within goal tolerance."""
        dx = abs(state.x - goal.x)
        dy = abs(state.y - goal.y)
        dtheta = abs(self._normalize_angle(state.theta - goal.theta))

        return (dx < self.config.goal_xy_tolerance and
                dy < self.config.goal_xy_tolerance and
                dtheta < self.config.goal_theta_tolerance)

    def _try_dubins_shortcut(self, node: Node, goal: State) -> List[State] | None:
        """
        Try to connect current node directly to goal via a Dubins curve.

        This is the "analytic expansion" — instead of searching grid cell by
        cell all the way to the goal, we try to draw a smooth curve directly
        to it. If the curve doesn't hit any obstacles, we're done instantly.

        Returns:
            Full path from start to goal if shortcut works, None otherwise.
        """
        radius = self.vehicle.config.min_turn_radius

        # Compute the shortest Dubins curve from here to the goal
        dubins_path = compute_dubins_path(node.state, goal, radius)
        if dubins_path is None:
            return None

        # Sample points along the curve and check each for collision
        sampled = dubins_path.sample(step_size=self.config.step_size * 0.5)
        for state in sampled:
            if not self.grid.in_bounds(state.x, state.y):
                return None
            if self._check_collision(state):
                return None

        # No collisions! Combine the A* path so far + the Dubins curve
        astar_path = self._reconstruct_path(node)
        # Skip the first Dubins point (it's the same as the last A* point)
        return astar_path + sampled[1:]

    def _reconstruct_path(self, node: Node) -> List[State]:
        """Reconstruct path by following parent pointers."""
        path = []
        current = node

        while current is not None:
            path.append(current.state)
            current = current.parent

        path.reverse()
        return path

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi


def smooth_path(path: List[State], iterations: int = 50, weight_data: float = 0.1, weight_smooth: float = 0.3) -> List[State]:
    """
    Smooth path using gradient descent.

    Args:
        path: Original path
        iterations: Number of smoothing iterations
        weight_data: Weight for staying close to original path
        weight_smooth: Weight for smoothness

    Returns:
        Smoothed path
    """
    if len(path) <= 2:
        return path

    # Copy path to arrays
    x = np.array([s.x for s in path], dtype=float)
    y = np.array([s.y for s in path], dtype=float)
    theta = np.array([s.theta for s in path], dtype=float)

    new_x = x.copy()
    new_y = y.copy()

    for _ in range(iterations):
        new_x[1:-1] += weight_data * (x[1:-1] - new_x[1:-1])
        new_x[1:-1] += weight_smooth * (new_x[:-2] + new_x[2:] - 2 * new_x[1:-1])

        new_y[1:-1] += weight_data * (y[1:-1] - new_y[1:-1])
        new_y[1:-1] += weight_smooth * (new_y[:-2] + new_y[2:] - 2 * new_y[1:-1])

    # Recompute headings
    new_theta = theta.copy()
    new_theta[:-1] = np.arctan2(new_y[1:] - new_y[:-1], new_x[1:] - new_x[:-1])
    new_theta[-1] = new_theta[-2]

    return [State(new_x[i], new_y[i], new_theta[i]) for i in range(len(path))]
