"""Tests for the occupancy grid."""

import pytest
import numpy as np

from src.grid import OccupancyGrid, GridConfig


class TestOccupancyGrid:
    """Tests for the OccupancyGrid collision and coordinate system."""

    def setup_method(self):
        """Create a small 20x20 meter grid before each test."""
        # WHY: A small grid keeps tests fast. 0.5m resolution = 40x40 cells.
        self.grid = OccupancyGrid(GridConfig(width=20.0, height=20.0, resolution=0.5))

    def test_empty_grid_no_collision(self):
        """An empty grid should have no obstacles anywhere."""
        # WHY: Baseline sanity check. If an empty map reports collisions,
        # the planner would think every path is blocked.
        assert not self.grid.is_occupied(10.0, 10.0)
        assert not self.grid.is_occupied(1.0, 1.0)

    def test_out_of_bounds_is_occupied(self):
        """Positions outside the map should count as occupied (blocked)."""
        # WHY: The car should never drive off the edge of the world.
        # Treating out-of-bounds as "wall" prevents that.
        assert self.grid.is_occupied(-1.0, 10.0)
        assert self.grid.is_occupied(10.0, -1.0)
        assert self.grid.is_occupied(25.0, 10.0)

    def test_in_bounds_check(self):
        """in_bounds should correctly identify valid positions."""
        # WHY: Used by the planner to skip moves that go off the map.
        assert self.grid.in_bounds(0.0, 0.0)
        assert self.grid.in_bounds(10.0, 10.0)
        assert not self.grid.in_bounds(-1.0, 5.0)
        assert not self.grid.in_bounds(5.0, 25.0)

    def test_add_circle_creates_obstacle(self):
        """Adding a circle should make its center occupied."""
        # WHY: Circles are used for things like pillars or trees.
        # The center must definitely be blocked.
        self.grid.add_circle(10.0, 10.0, radius=2.0)
        assert self.grid.is_occupied(10.0, 10.0)

    def test_add_circle_outside_is_free(self):
        """Points well outside a circle should remain free."""
        # WHY: The obstacle shouldn't "leak" beyond its radius.
        self.grid.add_circle(10.0, 10.0, radius=2.0)
        assert not self.grid.is_occupied(1.0, 1.0)

    def test_add_rectangle_creates_obstacle(self):
        """Adding a rectangle should make its center occupied."""
        # WHY: Rectangles represent walls, cars, buildings, etc.
        self.grid.add_rectangle(10.0, 10.0, width=4.0, height=4.0)
        assert self.grid.is_occupied(10.0, 10.0)

    def test_boundary_walls(self):
        """Adding boundaries should block the edges of the map."""
        # WHY: Boundaries keep the car from driving off the edge.
        self.grid.add_boundary(thickness=1.0)
        assert self.grid.is_occupied(0.1, 0.1)  # Corner — should be wall
        assert not self.grid.is_occupied(10.0, 10.0)  # Center — should be free

    def test_clearance_decreases_near_obstacle(self):
        """Clearance (distance to nearest wall) should be smaller near obstacles."""
        # WHY: The planner uses clearance to prefer paths that stay away from walls.
        # A point right next to an obstacle should have less clearance than a far point.
        self.grid.add_circle(10.0, 10.0, radius=2.0)
        close = self.grid.get_clearance(12.5, 10.0)  # Just outside the circle
        far = self.grid.get_clearance(18.0, 10.0)    # Far away
        assert far > close

    def test_world_to_grid_conversion(self):
        """World coordinates should convert to correct grid indices."""
        # WHY: A bug here means obstacles and collision checks would be in
        # the wrong place — the car would "see" walls that aren't there.
        gx, gy = self.grid.world_to_grid(5.0, 10.0)
        assert gx == 10  # 5.0 / 0.5 = 10
        assert gy == 20  # 10.0 / 0.5 = 20
