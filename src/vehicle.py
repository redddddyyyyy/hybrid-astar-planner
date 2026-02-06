"""Vehicle kinematic model for Hybrid A* planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class VehicleConfig:
    """Vehicle configuration parameters."""

    wheelbase: float = 2.5  # Distance between front and rear axles (m)
    width: float = 1.8  # Vehicle width (m)
    length: float = 4.5  # Vehicle length (m)
    max_steer: float = math.radians(35)  # Maximum steering angle (rad)
    min_turn_radius: float = 5.0  # Minimum turning radius (m)

    def __post_init__(self):
        # Compute minimum turn radius from wheelbase and max steering
        self.min_turn_radius = self.wheelbase / math.tan(self.max_steer)


@dataclass
class State:
    """Vehicle state: position and heading."""

    x: float
    y: float
    theta: float  # Heading angle (rad)

    def __iter__(self):
        return iter((self.x, self.y, self.theta))

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.theta)

    def distance_to(self, other: State) -> float:
        """Euclidean distance to another state."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class Vehicle:
    """
    Bicycle kinematic model for non-holonomic vehicle.

    Uses the bicycle model where the vehicle is represented by
    a single front wheel and single rear wheel on the centerline.
    """

    def __init__(self, config: VehicleConfig | None = None):
        self.config = config or VehicleConfig()

    def step(
        self,
        state: State,
        steer: float,
        distance: float,
        reverse: bool = False
    ) -> State:
        """
        Compute next state using bicycle kinematics.

        Args:
            state: Current vehicle state
            steer: Steering angle (rad), positive = left turn
            distance: Distance to travel (m)
            reverse: If True, move backward

        Returns:
            New vehicle state after motion
        """
        # Clamp steering angle
        steer = np.clip(steer, -self.config.max_steer, self.config.max_steer)

        # Direction multiplier for reverse
        direction = -1.0 if reverse else 1.0

        if abs(steer) < 1e-6:
            # Straight line motion
            new_x = state.x + direction * distance * math.cos(state.theta)
            new_y = state.y + direction * distance * math.sin(state.theta)
            new_theta = state.theta
        else:
            # Arc motion using bicycle model
            # Turn radius from rear axle
            turn_radius = self.config.wheelbase / math.tan(steer)

            # Angular change
            beta = direction * distance / turn_radius

            # New position (computed from instantaneous center of rotation)
            cx = state.x - turn_radius * math.sin(state.theta)
            cy = state.y + turn_radius * math.cos(state.theta)

            new_theta = state.theta + beta
            new_x = cx + turn_radius * math.sin(new_theta)
            new_y = cy - turn_radius * math.cos(new_theta)

        # Normalize angle to [-pi, pi]
        new_theta = self._normalize_angle(new_theta)

        return State(new_x, new_y, new_theta)

    def get_motion_primitives(
        self,
        num_angles: int = 5,
        step_size: float = 1.5,
        include_reverse: bool = True
    ) -> List[Tuple[float, float, bool]]:
        """
        Generate discrete motion primitives.

        Args:
            num_angles: Number of steering angles to sample
            step_size: Distance for each primitive (m)
            include_reverse: Include reverse motions

        Returns:
            List of (steering_angle, distance, reverse) tuples
        """
        primitives = []

        # Sample steering angles from -max to +max
        steers = np.linspace(
            -self.config.max_steer,
            self.config.max_steer,
            num_angles
        )

        # Forward motions
        for steer in steers:
            primitives.append((steer, step_size, False))

        # Reverse motions
        if include_reverse:
            for steer in steers:
                primitives.append((steer, step_size, True))

        return primitives

    def get_footprint(self, state: State) -> np.ndarray:
        """
        Get vehicle footprint corners at given state.

        Returns:
            4x2 array of corner positions [[x1,y1], [x2,y2], ...]
        """
        L = self.config.length
        W = self.config.width

        # Vehicle corners relative to rear axle center
        # [front-left, front-right, rear-right, rear-left]
        corners_local = np.array([
            [L * 0.75, W / 2],   # Front left
            [L * 0.75, -W / 2],  # Front right
            [-L * 0.25, -W / 2], # Rear right
            [-L * 0.25, W / 2],  # Rear left
        ])

        # Rotation matrix
        c, s = math.cos(state.theta), math.sin(state.theta)
        R = np.array([[c, -s], [s, c]])

        # Transform to world frame
        corners_world = corners_local @ R.T + np.array([state.x, state.y])

        return corners_world

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
