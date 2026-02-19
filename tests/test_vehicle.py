"""Tests for the vehicle kinematic model."""

import math
import pytest
import numpy as np

from src.vehicle import Vehicle, VehicleConfig, State


class TestState:
    """Tests for the State dataclass."""

    def test_distance_to_same_point(self):
        """Distance from a point to itself should be zero."""
        # WHY: Sanity check — if this fails, all distance math is broken.
        s = State(5.0, 5.0, 0.0)
        assert s.distance_to(s) == 0.0

    def test_distance_to_other_point(self):
        """Classic 3-4-5 triangle should give distance of 5."""
        # WHY: Uses a known right triangle to verify the Euclidean formula.
        # If you remember from math class: 3² + 4² = 5²
        a = State(0.0, 0.0, 0.0)
        b = State(3.0, 4.0, 0.0)
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_to_tuple(self):
        """State should unpack to (x, y, theta) tuple."""
        s = State(1.0, 2.0, 0.5)
        assert s.to_tuple() == (1.0, 2.0, 0.5)


class TestVehicle:
    """Tests for the Vehicle kinematic model."""

    def setup_method(self):
        """Create a default vehicle before each test."""
        # WHY: Every test needs a car, so we make one here instead of
        # repeating it in every single test function.
        self.vehicle = Vehicle()

    def test_straight_line_forward(self):
        """Driving straight with zero steering should move along heading."""
        # WHY: The simplest possible move. If this fails, nothing else works.
        # Start at origin, facing right (theta=0), drive 5 meters.
        # Should end up at (5, 0).
        start = State(0.0, 0.0, 0.0)
        result = self.vehicle.step(start, steer=0.0, distance=5.0)

        assert result.x == pytest.approx(5.0, abs=1e-6)
        assert result.y == pytest.approx(0.0, abs=1e-6)
        assert result.theta == pytest.approx(0.0, abs=1e-6)

    def test_straight_line_reverse(self):
        """Reversing straight should move backward along heading."""
        # WHY: Reverse is used in parking scenarios. The car should go
        # backward (negative x direction when facing right).
        start = State(5.0, 0.0, 0.0)
        result = self.vehicle.step(start, steer=0.0, distance=3.0, reverse=True)

        assert result.x == pytest.approx(2.0, abs=1e-6)
        assert result.y == pytest.approx(0.0, abs=1e-6)

    def test_left_turn_moves_left(self):
        """Positive steering (left turn) should move the car leftward (positive y)."""
        # WHY: Verifies the steering direction convention is correct.
        # A left turn from facing right should curve upward on the map.
        start = State(0.0, 0.0, 0.0)
        result = self.vehicle.step(start, steer=0.3, distance=2.0)

        assert result.y > 0  # Car moved upward (left turn)
        assert result.theta > 0  # Heading rotated counter-clockwise

    def test_right_turn_moves_right(self):
        """Negative steering (right turn) should move the car rightward (negative y)."""
        # WHY: Mirror of the left turn test. Both directions must work.
        start = State(0.0, 0.0, 0.0)
        result = self.vehicle.step(start, steer=-0.3, distance=2.0)

        assert result.y < 0  # Car moved downward (right turn)
        assert result.theta < 0  # Heading rotated clockwise

    def test_steering_is_clamped(self):
        """Steering beyond max_steer should be clamped."""
        # WHY: A real steering wheel has limits. Passing steer=999 shouldn't
        # break the math — it should just use max_steer instead.
        start = State(0.0, 0.0, 0.0)
        huge_steer = self.vehicle.step(start, steer=999.0, distance=2.0)
        max_steer = self.vehicle.step(start, steer=self.vehicle.config.max_steer, distance=2.0)

        assert huge_steer.x == pytest.approx(max_steer.x, abs=1e-6)
        assert huge_steer.y == pytest.approx(max_steer.y, abs=1e-6)

    def test_normalize_angle(self):
        """Angles should always be normalized to [-pi, pi]."""
        # WHY: This is the fix we made in Step 1! Let's prove it works.
        assert Vehicle._normalize_angle(0) == pytest.approx(0)
        assert Vehicle._normalize_angle(2 * math.pi) == pytest.approx(0, abs=1e-6)
        # -pi and pi are the same direction; our formula returns -pi (range is [-pi, pi))
        assert abs(Vehicle._normalize_angle(-3 * math.pi)) == pytest.approx(math.pi, abs=1e-6)
        assert Vehicle._normalize_angle(100 * math.pi) == pytest.approx(0, abs=1e-6)

    def test_footprint_has_four_corners(self):
        """Vehicle footprint should always be a rectangle (4 corners)."""
        # WHY: The collision checker uses this shape. Wrong number of corners
        # means wrong collision results.
        state = State(10.0, 10.0, 0.5)
        footprint = self.vehicle.get_footprint(state)

        assert footprint.shape == (4, 2)

    def test_footprint_centered_on_vehicle(self):
        """Footprint center should be near the vehicle position."""
        # WHY: If the footprint drifts away from the actual car position,
        # collisions would be checked in the wrong place.
        state = State(10.0, 10.0, 0.0)
        footprint = self.vehicle.get_footprint(state)
        center = footprint.mean(axis=0)

        # Center should be close to vehicle position (offset by rear axle placement)
        assert abs(center[0] - state.x) < self.vehicle.config.length
        assert abs(center[1] - state.y) < self.vehicle.config.width

    def test_motion_primitives_count(self):
        """Should generate correct number of motion primitives."""
        # WHY: The planner tries every primitive each iteration. Wrong count
        # means missing moves or wasted computation.
        prims = self.vehicle.get_motion_primitives(num_angles=5, include_reverse=True)
        assert len(prims) == 10  # 5 forward + 5 reverse

        prims_no_rev = self.vehicle.get_motion_primitives(num_angles=5, include_reverse=False)
        assert len(prims_no_rev) == 5  # 5 forward only
