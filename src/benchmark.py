"""
Benchmark comparison of path planning algorithms.

Compares:
    - A* (grid-based, holonomic)
    - RRT* (sampling-based)
    - Hybrid A* (grid-based, non-holonomic)

Metrics:
    - Path length
    - Computation time
    - Path smoothness
    - Success rate
"""

from __future__ import annotations

import time
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
import heapq

import numpy as np

from .vehicle import State, Vehicle
from .grid import OccupancyGrid
from .hybrid_astar import HybridAStar, PlannerConfig, smooth_path


@dataclass
class BenchmarkResult:
    """Results from a single planning run."""

    algorithm: str
    success: bool
    path_length: float = 0.0
    computation_time: float = 0.0
    num_nodes_expanded: int = 0
    path_smoothness: float = 0.0  # Average curvature
    path: List[Tuple[float, float, float]] = field(default_factory=list)

    def __str__(self) -> str:
        if not self.success:
            return f"{self.algorithm}: FAILED"
        return (f"{self.algorithm}: length={self.path_length:.2f}m, "
                f"time={self.computation_time*1000:.1f}ms, "
                f"nodes={self.num_nodes_expanded}")


class Planner(ABC):
    """Abstract base class for planners."""

    @abstractmethod
    def plan(self, start: Tuple[float, float, float],
             goal: Tuple[float, float, float]) -> List[Tuple[float, float, float]] | None:
        pass

    @property
    @abstractmethod
    def nodes_expanded(self) -> int:
        pass


class AStarPlanner(Planner):
    """
    Classic A* on a 2D grid (holonomic, ignores heading).

    This serves as a baseline - it finds the shortest geometric path
    but ignores vehicle kinematics.
    """

    def __init__(self, grid: OccupancyGrid):
        self.grid = grid
        self._nodes_expanded = 0

    @property
    def nodes_expanded(self) -> int:
        return self._nodes_expanded

    def plan(self, start: Tuple[float, float, float],
             goal: Tuple[float, float, float]) -> List[Tuple[float, float, float]] | None:
        """Plan using A* on 2D grid."""
        self._nodes_expanded = 0

        # Convert to grid coordinates
        start_grid = self.grid.world_to_grid(start[0], start[1])
        goal_grid = self.grid.world_to_grid(goal[0], goal[1])

        # A* search
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}

        # 8-connected neighbors
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1),  (1, 0), (1, 1)
        ]

        while open_set:
            _, current = heapq.heappop(open_set)
            self._nodes_expanded += 1

            if current == goal_grid:
                # Reconstruct path
                path = self._reconstruct_path(came_from, current, start, goal)
                return path

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check bounds
                if not (0 <= neighbor[0] < self.grid.config.x_cells and
                        0 <= neighbor[1] < self.grid.config.y_cells):
                    continue

                # Check collision
                if self.grid.grid[neighbor[1], neighbor[0]]:
                    continue

                # Cost (diagonal = sqrt(2))
                move_cost = math.sqrt(dx*dx + dy*dy) * self.grid.config.resolution
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    # Heuristic: Euclidean distance
                    h = math.sqrt((neighbor[0] - goal_grid[0])**2 +
                                 (neighbor[1] - goal_grid[1])**2) * self.grid.config.resolution
                    f = tentative_g + h

                    heapq.heappush(open_set, (f, neighbor))

        return None  # No path found

    def _reconstruct_path(self, came_from, current, start, goal):
        """Reconstruct path from A* search."""
        path = []
        while current in came_from:
            wx, wy = self.grid.grid_to_world(current[0], current[1])
            path.append((wx, wy, 0.0))  # No heading for A*
            current = came_from[current]

        # Add start
        wx, wy = self.grid.grid_to_world(current[0], current[1])
        path.append((wx, wy, start[2]))

        path.reverse()

        # Compute headings from path direction
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            theta = math.atan2(dy, dx)
            path[i] = (path[i][0], path[i][1], theta)

        # Set goal heading
        path[-1] = (path[-1][0], path[-1][1], goal[2])

        return path


class RRTStarPlanner(Planner):
    """
    RRT* (Rapidly-exploring Random Tree Star) planner.

    Sampling-based planner that builds a tree of collision-free paths
    and optimizes for shortest path length.
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        max_iterations: int = 5000,
        step_size: float = 2.0,
        goal_sample_rate: float = 0.1,
        search_radius: float = 5.0
    ):
        self.grid = grid
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self._nodes_expanded = 0

    @property
    def nodes_expanded(self) -> int:
        return self._nodes_expanded

    def plan(self, start: Tuple[float, float, float],
             goal: Tuple[float, float, float]) -> List[Tuple[float, float, float]] | None:
        """Plan using RRT*."""
        self._nodes_expanded = 0

        # Node structure: (x, y, parent_idx, cost)
        nodes = [(start[0], start[1], -1, 0.0)]
        goal_threshold = 1.0

        for _ in range(self.max_iterations):
            self._nodes_expanded += 1

            # Sample random point (bias toward goal)
            if random.random() < self.goal_sample_rate:
                sample = (goal[0], goal[1])
            else:
                sample = (
                    random.uniform(0, self.grid.config.width),
                    random.uniform(0, self.grid.config.height)
                )

            # Find nearest node
            nearest_idx = self._nearest(nodes, sample)
            nearest = nodes[nearest_idx]

            # Steer toward sample
            new_point = self._steer(nearest, sample)

            # Check collision
            if self._collision_free(nearest, new_point):
                # Find nearby nodes for rewiring
                near_indices = self._near(nodes, new_point)

                # Choose best parent
                min_cost = nearest[3] + self._distance(nearest, new_point)
                min_idx = nearest_idx

                for idx in near_indices:
                    node = nodes[idx]
                    new_cost = node[3] + self._distance(node, new_point)
                    if new_cost < min_cost and self._collision_free(node, new_point):
                        min_cost = new_cost
                        min_idx = idx

                # Add new node
                new_node = (new_point[0], new_point[1], min_idx, min_cost)
                new_idx = len(nodes)
                nodes.append(new_node)

                # Rewire nearby nodes
                for idx in near_indices:
                    node = nodes[idx]
                    new_cost = min_cost + self._distance(new_point, node)
                    if new_cost < node[3] and self._collision_free(new_point, node):
                        nodes[idx] = (node[0], node[1], new_idx, new_cost)

                # Check if goal reached
                if self._distance(new_point, goal) < goal_threshold:
                    return self._reconstruct_path(nodes, len(nodes) - 1, start, goal)

        return None  # No path found

    def _nearest(self, nodes, point):
        """Find nearest node to point."""
        min_dist = float('inf')
        min_idx = 0
        for i, node in enumerate(nodes):
            d = self._distance(node, point)
            if d < min_dist:
                min_dist = d
                min_idx = i
        return min_idx

    def _near(self, nodes, point):
        """Find all nodes within search radius."""
        return [i for i, node in enumerate(nodes)
                if self._distance(node, point) < self.search_radius]

    def _steer(self, from_node, to_point):
        """Steer from node toward point by step_size."""
        dx = to_point[0] - from_node[0]
        dy = to_point[1] - from_node[1]
        d = math.sqrt(dx*dx + dy*dy)

        if d < self.step_size:
            return to_point

        ratio = self.step_size / d
        return (from_node[0] + dx * ratio, from_node[1] + dy * ratio)

    def _distance(self, p1, p2):
        """Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _collision_free(self, p1, p2):
        """Check if path between two points is collision-free."""
        d = self._distance(p1, p2)
        num_checks = max(2, int(d / self.grid.config.resolution))

        for i in range(num_checks + 1):
            t = i / num_checks
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if self.grid.is_occupied(x, y):
                return False

        return True

    def _reconstruct_path(self, nodes, goal_idx, start, goal):
        """Reconstruct path from tree."""
        path = []
        idx = goal_idx

        while idx >= 0:
            node = nodes[idx]
            path.append((node[0], node[1], 0.0))
            idx = node[2]

        path.reverse()

        # Compute headings
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            theta = math.atan2(dy, dx)
            path[i] = (path[i][0], path[i][1], theta)

        path[-1] = (path[-1][0], path[-1][1], goal[2])

        return path


class HybridAStarWrapper(Planner):
    """Wrapper for Hybrid A* planner to match interface."""

    def __init__(self, grid: OccupancyGrid, vehicle: Vehicle = None):
        self.grid = grid
        self.vehicle = vehicle or Vehicle()
        self._nodes_expanded = 0

    @property
    def nodes_expanded(self) -> int:
        return self._nodes_expanded

    def plan(self, start: Tuple[float, float, float],
             goal: Tuple[float, float, float]) -> List[Tuple[float, float, float]] | None:
        """Plan using Hybrid A*."""
        planner = HybridAStar(self.grid, self.vehicle)

        start_state = State(*start)
        goal_state = State(*goal)

        path = planner.plan(start_state, goal_state)

        if path is None:
            return None

        # Smooth path
        path = smooth_path(path)

        # Convert to tuples
        return [(s.x, s.y, s.theta) for s in path]


def compute_path_length(path: List[Tuple[float, float, float]]) -> float:
    """Compute total path length."""
    if len(path) < 2:
        return 0.0

    length = 0.0
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        length += math.sqrt(dx*dx + dy*dy)

    return length


def compute_path_smoothness(path: List[Tuple[float, float, float]]) -> float:
    """
    Compute path smoothness as average curvature.

    Lower values indicate smoother paths.
    """
    if len(path) < 3:
        return 0.0

    curvatures = []
    for i in range(1, len(path) - 1):
        # Approximate curvature using angle change
        dx1 = path[i][0] - path[i-1][0]
        dy1 = path[i][1] - path[i-1][1]
        dx2 = path[i+1][0] - path[i][0]
        dy2 = path[i+1][1] - path[i][1]

        theta1 = math.atan2(dy1, dx1)
        theta2 = math.atan2(dy2, dx2)

        dtheta = abs(theta2 - theta1)
        if dtheta > math.pi:
            dtheta = 2 * math.pi - dtheta

        curvatures.append(dtheta)

    return np.mean(curvatures) if curvatures else 0.0


class Benchmark:
    """
    Benchmark suite for comparing path planning algorithms.
    """

    def __init__(self, grid: OccupancyGrid, vehicle: Vehicle = None):
        self.grid = grid
        self.vehicle = vehicle or Vehicle()

        # Initialize planners
        self.planners: Dict[str, Planner] = {
            'A*': AStarPlanner(grid),
            'RRT*': RRTStarPlanner(grid),
            'Hybrid A*': HybridAStarWrapper(grid, self.vehicle),
        }

    def run_single(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        algorithm: str
    ) -> BenchmarkResult:
        """Run a single planning benchmark."""
        if algorithm not in self.planners:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        planner = self.planners[algorithm]

        # Time the planning
        start_time = time.perf_counter()
        path = planner.plan(start, goal)
        elapsed = time.perf_counter() - start_time

        if path is None:
            return BenchmarkResult(
                algorithm=algorithm,
                success=False,
                computation_time=elapsed,
                num_nodes_expanded=planner.nodes_expanded
            )

        return BenchmarkResult(
            algorithm=algorithm,
            success=True,
            path_length=compute_path_length(path),
            computation_time=elapsed,
            num_nodes_expanded=planner.nodes_expanded,
            path_smoothness=compute_path_smoothness(path),
            path=path
        )

    def run_comparison(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float]
    ) -> Dict[str, BenchmarkResult]:
        """Run all planners and compare results."""
        results = {}
        for name in self.planners:
            results[name] = self.run_single(start, goal, name)
        return results

    def run_benchmark_suite(
        self,
        scenarios: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
        num_trials: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Run comprehensive benchmark across multiple scenarios.

        Returns aggregated statistics.
        """
        stats = {name: {
            'success_rate': 0.0,
            'avg_length': 0.0,
            'avg_time': 0.0,
            'avg_nodes': 0.0,
            'avg_smoothness': 0.0
        } for name in self.planners}

        total_runs = len(scenarios) * num_trials

        for start, goal in scenarios:
            for _ in range(num_trials):
                results = self.run_comparison(start, goal)

                for name, result in results.items():
                    if result.success:
                        stats[name]['success_rate'] += 1 / total_runs
                        stats[name]['avg_length'] += result.path_length / total_runs
                        stats[name]['avg_time'] += result.computation_time / total_runs
                        stats[name]['avg_nodes'] += result.nodes_expanded / total_runs
                        stats[name]['avg_smoothness'] += result.path_smoothness / total_runs

        return stats

    def print_comparison(self, results: Dict[str, BenchmarkResult]) -> None:
        """Print comparison table."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        print(f"{'Algorithm':<15} {'Success':<10} {'Length (m)':<12} {'Time (ms)':<12} {'Nodes':<10} {'Smoothness':<10}")
        print("-" * 70)

        for name, result in results.items():
            if result.success:
                print(f"{name:<15} {'YES':<10} {result.path_length:<12.2f} "
                      f"{result.computation_time*1000:<12.1f} {result.num_nodes_expanded:<10} "
                      f"{result.path_smoothness:<10.4f}")
            else:
                print(f"{name:<15} {'NO':<10} {'-':<12} "
                      f"{result.computation_time*1000:<12.1f} {result.num_nodes_expanded:<10} {'-':<10}")

        print("=" * 70)
