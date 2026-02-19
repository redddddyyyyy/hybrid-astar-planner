"""Tests for the Hybrid A* planner."""

import math
import pytest

from src.vehicle import Vehicle, State
from src.grid import OccupancyGrid, GridConfig
from src.hybrid_astar import HybridAStar, PlannerConfig, smooth_path


class TestHybridAStar:
    """Tests for the Hybrid A* path planning algorithm."""

    def setup_method(self):
        """Create a simple open grid and planner before each test."""
        # WHY: Most tests need a basic planner. We make a small 30x30m grid
        # with no obstacles — individual tests add obstacles as needed.
        self.grid = OccupancyGrid(GridConfig(width=30.0, height=30.0, resolution=0.5))
        self.vehicle = Vehicle()
        self.planner = HybridAStar(self.grid, self.vehicle)

    def test_straight_path_no_obstacles(self):
        """Planner should find a path in an open field."""
        # WHY: The simplest possible scenario. If the planner can't find a
        # path with ZERO obstacles, something is fundamentally broken.
        start = State(5.0, 15.0, 0.0)
        goal = State(25.0, 15.0, 0.0)

        path = self.planner.plan(start, goal)

        assert path is not None
        assert len(path) >= 2  # At least start and goal

    def test_path_starts_near_start(self):
        """First waypoint should be at the start position."""
        # WHY: A path that doesn't begin where the car actually is
        # would mean the car has to teleport.
        start = State(5.0, 15.0, 0.0)
        goal = State(25.0, 15.0, 0.0)

        path = self.planner.plan(start, goal)

        assert path is not None
        assert path[0].x == pytest.approx(start.x)
        assert path[0].y == pytest.approx(start.y)

    def test_path_ends_near_goal(self):
        """Last waypoint should be close to the goal position."""
        # WHY: A path that doesn't reach the destination is useless.
        # We check "close to" because there's a goal tolerance.
        start = State(5.0, 15.0, 0.0)
        goal = State(25.0, 15.0, 0.0)
        config = PlannerConfig(goal_xy_tolerance=1.5)
        planner = HybridAStar(self.grid, self.vehicle, config)

        path = planner.plan(start, goal)

        assert path is not None
        assert abs(path[-1].x - goal.x) < config.goal_xy_tolerance
        assert abs(path[-1].y - goal.y) < config.goal_xy_tolerance

    def test_path_around_obstacle(self):
        """Planner should route around an obstacle instead of through it."""
        # WHY: This is the whole point of a path planner! Put a wall between
        # start and goal — the path must go around it, not through it.
        self.grid.add_rectangle(15.0, 15.0, width=2.0, height=20.0)

        start = State(5.0, 15.0, 0.0)
        goal = State(25.0, 15.0, 0.0)

        path = self.planner.plan(start, goal)

        assert path is not None
        # Path should be longer than a straight line (20m) because it detours
        total_dist = sum(
            path[i].distance_to(path[i + 1]) for i in range(len(path) - 1)
        )
        assert total_dist > 20.0

    def test_no_path_when_goal_blocked(self):
        """Planner should return None when goal is inside an obstacle."""
        # WHY: If the destination is literally inside a wall, there's no
        # valid path. The planner should say "impossible" not crash.
        self.grid.add_circle(25.0, 15.0, radius=3.0)

        start = State(5.0, 15.0, 0.0)
        goal = State(25.0, 15.0, 0.0)  # Inside the circle

        path = self.planner.plan(start, goal)

        assert path is None

    def test_no_path_when_start_blocked(self):
        """Planner should return None when start is inside an obstacle."""
        # WHY: If the car is already stuck inside a wall, we can't plan.
        self.grid.add_circle(5.0, 15.0, radius=3.0)

        start = State(5.0, 15.0, 0.0)  # Inside the circle
        goal = State(25.0, 15.0, 0.0)

        path = self.planner.plan(start, goal)

        assert path is None

    def test_reverse_motion_enabled(self):
        """With reverse enabled, planner should have both forward and reverse primitives."""
        # WHY: Parking scenarios need reverse. If the planner can't reverse,
        # it can't parallel park or back out of tight spaces.
        config = PlannerConfig(include_reverse=True, num_steer_angles=3)
        planner = HybridAStar(self.grid, self.vehicle, config)

        # 3 forward + 3 reverse = 6 primitives
        assert len(planner.primitives) == 6

    def test_reverse_motion_disabled(self):
        """With reverse disabled, planner should only have forward primitives."""
        # WHY: Some vehicles (like bikes) can't reverse. The planner should
        # respect that constraint.
        config = PlannerConfig(include_reverse=False, num_steer_angles=3)
        planner = HybridAStar(self.grid, self.vehicle, config)

        assert len(planner.primitives) == 3


class TestSmoothPath:
    """Tests for the path smoothing function."""

    def test_short_path_unchanged(self):
        """A path with 2 or fewer points can't be smoothed."""
        # WHY: You need at least 3 points to smooth (a middle point with
        # two neighbors). With 2 points it's already a straight line.
        path = [State(0, 0, 0), State(5, 0, 0)]
        result = smooth_path(path)
        assert len(result) == 2

    def test_smoothed_path_same_length(self):
        """Smoothing shouldn't add or remove waypoints."""
        # WHY: Smoothing moves points, it doesn't create or delete them.
        # The path should have the same number of waypoints before and after.
        path = [State(i, 0, 0) for i in range(10)]
        result = smooth_path(path, iterations=50)
        assert len(result) == len(path)

    def test_endpoints_preserved(self):
        """Smoothing should NOT move the start and goal points."""
        # WHY: The car must still start and end at the correct positions.
        # Smoothing only adjusts the middle waypoints.
        path = [State(0, 0, 0), State(3, 5, 0.5), State(6, 1, 0.2), State(10, 0, 0)]
        result = smooth_path(path, iterations=100)

        assert result[0].x == pytest.approx(0.0)
        assert result[0].y == pytest.approx(0.0)
        assert result[-1].x == pytest.approx(10.0)
        assert result[-1].y == pytest.approx(0.0)

    def test_smoothing_reduces_jaggedness(self):
        """A zigzag path should become smoother after smoothing."""
        # WHY: This is literally what smoothing is for. A zigzag like
        # up-down-up-down should become more like a gentle wave.
        path = [
            State(0, 0, 0),
            State(2, 3, 0),   # Up
            State(4, -2, 0),  # Down
            State(6, 4, 0),   # Up
            State(8, -1, 0),  # Down
            State(10, 0, 0),
        ]

        result = smooth_path(path, iterations=200)

        # After smoothing, the peaks should be less extreme
        original_range = max(s.y for s in path) - min(s.y for s in path)
        smoothed_range = max(s.y for s in result) - min(s.y for s in result)

        assert smoothed_range < original_range
