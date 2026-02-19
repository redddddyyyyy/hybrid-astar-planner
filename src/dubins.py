"""
Dubins path computation for non-holonomic vehicles.

Dubins paths are the shortest paths for a vehicle that can only move
forward with a minimum turning radius. Each path consists of at most
3 segments: combinations of Left turns, Right turns, and Straight lines.

The 6 possible Dubins path types are:
    LSL, LSR, RSL, RSR, RLR, LRL

Reference:
    Dubins, L.E. (1957). "On Curves of Minimal Length with a Constraint
    on Average Curvature, and with Prescribed Initial and Terminal
    Positions and Tangents"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np

from .vehicle import State


class SegmentType(Enum):
    """Dubins path segment types."""
    LEFT = 'L'
    STRAIGHT = 'S'
    RIGHT = 'R'


class DubinsPathType(Enum):
    """The 6 Dubins path types."""
    LSL = (SegmentType.LEFT, SegmentType.STRAIGHT, SegmentType.LEFT)
    LSR = (SegmentType.LEFT, SegmentType.STRAIGHT, SegmentType.RIGHT)
    RSL = (SegmentType.RIGHT, SegmentType.STRAIGHT, SegmentType.LEFT)
    RSR = (SegmentType.RIGHT, SegmentType.STRAIGHT, SegmentType.RIGHT)
    RLR = (SegmentType.RIGHT, SegmentType.LEFT, SegmentType.RIGHT)
    LRL = (SegmentType.LEFT, SegmentType.RIGHT, SegmentType.LEFT)


@dataclass
class DubinsPath:
    """Represents a complete Dubins path."""

    path_type: DubinsPathType
    lengths: Tuple[float, float, float]  # Segment lengths (normalized by radius)
    radius: float  # Turning radius
    start: State
    goal: State

    @property
    def total_length(self) -> float:
        """Total path length in world units."""
        return sum(self.lengths) * self.radius

    def sample(self, step_size: float = 0.1) -> List[State]:
        """
        Sample points along the Dubins path.

        Args:
            step_size: Distance between samples (meters)

        Returns:
            List of states along the path
        """
        points = []
        segments = self.path_type.value

        # Current state
        x, y, theta = self.start.x, self.start.y, self.start.theta

        for seg_type, seg_len in zip(segments, self.lengths):
            # Number of samples for this segment
            arc_length = seg_len * self.radius
            num_samples = max(1, int(arc_length / step_size))

            for i in range(num_samples):
                # Add current point
                points.append(State(x, y, theta))

                # Step forward
                ds = arc_length / num_samples

                if seg_type == SegmentType.STRAIGHT:
                    x += ds * math.cos(theta)
                    y += ds * math.sin(theta)
                elif seg_type == SegmentType.LEFT:
                    dtheta = ds / self.radius
                    # Arc center is to the left
                    cx = x - self.radius * math.sin(theta)
                    cy = y + self.radius * math.cos(theta)
                    theta += dtheta
                    x = cx + self.radius * math.sin(theta)
                    y = cy - self.radius * math.cos(theta)
                else:  # RIGHT
                    dtheta = ds / self.radius
                    # Arc center is to the right
                    cx = x + self.radius * math.sin(theta)
                    cy = y - self.radius * math.cos(theta)
                    theta -= dtheta
                    x = cx - self.radius * math.sin(theta)
                    y = cy + self.radius * math.cos(theta)

                theta = _normalize_angle(theta)

        # Add final point
        points.append(State(self.goal.x, self.goal.y, self.goal.theta))

        return points


def compute_dubins_path(
    start: State,
    goal: State,
    radius: float
) -> DubinsPath | None:
    """
    Compute the shortest Dubins path between two states.

    Args:
        start: Starting state (x, y, theta)
        goal: Goal state (x, y, theta)
        radius: Minimum turning radius

    Returns:
        Shortest DubinsPath, or None if no valid path exists
    """
    # Normalize problem to unit turning radius
    dx = goal.x - start.x
    dy = goal.y - start.y
    d = math.sqrt(dx * dx + dy * dy) / radius

    # Angle from start to goal
    theta = math.atan2(dy, dx)

    # Relative heading angles (normalized)
    alpha = _normalize_angle(start.theta - theta)
    beta = _normalize_angle(goal.theta - theta)

    # Try all path types and find shortest
    best_path = None
    best_length = float('inf')

    path_functions = [
        (_dubins_LSL, DubinsPathType.LSL),
        (_dubins_LSR, DubinsPathType.LSR),
        (_dubins_RSL, DubinsPathType.RSL),
        (_dubins_RSR, DubinsPathType.RSR),
        (_dubins_RLR, DubinsPathType.RLR),
        (_dubins_LRL, DubinsPathType.LRL),
    ]

    for func, path_type in path_functions:
        lengths = func(alpha, beta, d)
        if lengths is not None:
            total = sum(lengths)
            if total < best_length:
                best_length = total
                best_path = DubinsPath(
                    path_type=path_type,
                    lengths=lengths,
                    radius=radius,
                    start=start,
                    goal=goal
                )

    return best_path


def _dubins_LSL(alpha: float, beta: float, d: float) -> Tuple[float, float, float] | None:
    """Compute LSL path lengths."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)

    p_sq = 2 + d * d - 2 * (ca * cb + sa * sb - d * (sa - sb))
    if p_sq < 0:
        return None

    tmp = math.atan2(cb - ca, d + sa - sb)
    t = _normalize_angle(-alpha + tmp)
    p = math.sqrt(p_sq)
    q = _normalize_angle(beta - tmp)

    if t < 0 or p < 0 or q < 0:
        return None

    return (t, p, q)


def _dubins_RSR(alpha: float, beta: float, d: float) -> Tuple[float, float, float] | None:
    """Compute RSR path lengths."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)

    p_sq = 2 + d * d - 2 * (ca * cb + sa * sb - d * (sb - sa))
    if p_sq < 0:
        return None

    tmp = math.atan2(ca - cb, d - sa + sb)
    t = _normalize_angle(alpha - tmp)
    p = math.sqrt(p_sq)
    q = _normalize_angle(-beta + tmp)

    if t < 0 or p < 0 or q < 0:
        return None

    return (t, p, q)


def _dubins_LSR(alpha: float, beta: float, d: float) -> Tuple[float, float, float] | None:
    """Compute LSR path lengths."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)

    p_sq = -2 + d * d + 2 * (ca * cb + sa * sb + d * (sa + sb))
    if p_sq < 0:
        return None

    p = math.sqrt(p_sq)
    tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2, p)
    t = _normalize_angle(-alpha + tmp)
    q = _normalize_angle(-beta + tmp)

    if t < 0 or p < 0 or q < 0:
        return None

    return (t, p, q)


def _dubins_RSL(alpha: float, beta: float, d: float) -> Tuple[float, float, float] | None:
    """Compute RSL path lengths."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)

    p_sq = -2 + d * d + 2 * (ca * cb + sa * sb - d * (sa + sb))
    if p_sq < 0:
        return None

    p = math.sqrt(p_sq)
    tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2, p)
    t = _normalize_angle(alpha - tmp)
    q = _normalize_angle(beta - tmp)

    if t < 0 or p < 0 or q < 0:
        return None

    return (t, p, q)


def _dubins_RLR(alpha: float, beta: float, d: float) -> Tuple[float, float, float] | None:
    """Compute RLR path lengths."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)

    tmp = (6 - d * d + 2 * (ca * cb + sa * sb + d * (sa - sb))) / 8
    if abs(tmp) > 1:
        return None

    p = _normalize_angle(2 * math.pi - math.acos(tmp))
    t = _normalize_angle(alpha - math.atan2(ca - cb, d - sa + sb) + p / 2)
    q = _normalize_angle(alpha - beta - t + p)

    if t < 0 or p < 0 or q < 0:
        return None

    return (t, p, q)


def _dubins_LRL(alpha: float, beta: float, d: float) -> Tuple[float, float, float] | None:
    """Compute LRL path lengths."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)

    tmp = (6 - d * d + 2 * (ca * cb + sa * sb - d * (sa - sb))) / 8
    if abs(tmp) > 1:
        return None

    p = _normalize_angle(2 * math.pi - math.acos(tmp))
    t = _normalize_angle(-alpha + math.atan2(-ca + cb, d + sa - sb) + p / 2)
    q = _normalize_angle(beta - alpha - t + p)

    if t < 0 or p < 0 or q < 0:
        return None

    return (t, p, q)


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [0, 2*pi)."""
    return angle % (2 * math.pi)


class DubinsPlanner:
    """
    Path planner using Dubins curves for goal connection.

    Combines grid-based search with Dubins curves for
    smooth, kinematically feasible paths.
    """

    def __init__(self, turn_radius: float = 5.0):
        self.turn_radius = turn_radius

    def connect(self, start: State, goal: State) -> List[State] | None:
        """
        Compute Dubins path between two states.

        Args:
            start: Starting state
            goal: Goal state

        Returns:
            List of states along the Dubins path, or None if no path exists
        """
        path = compute_dubins_path(start, goal, self.turn_radius)
        if path is None:
            return None

        return path.sample(step_size=0.2)

    def path_length(self, start: State, goal: State) -> float:
        """Compute Dubins path length (useful as heuristic)."""
        path = compute_dubins_path(start, goal, self.turn_radius)
        if path is None:
            return float('inf')
        return path.total_length


def visualize_dubins_path(
    start: State,
    goal: State,
    radius: float,
    ax=None
) -> None:
    """
    Visualize a Dubins path.

    Args:
        start: Starting state
        goal: Goal state
        radius: Turning radius
        ax: Matplotlib axes (creates new figure if None)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    path = compute_dubins_path(start, goal, radius)
    if path is None:
        print("No valid Dubins path found")
        return

    # Sample and plot path
    points = path.sample(step_size=0.1)
    x = [p.x for p in points]
    y = [p.y for p in points]

    ax.plot(x, y, 'b-', linewidth=2, label=f'{path.path_type.name} (L={path.total_length:.2f}m)')

    # Plot start and goal
    for state, color, label in [(start, 'green', 'Start'), (goal, 'red', 'Goal')]:
        ax.plot(state.x, state.y, 'o', color=color, markersize=10)
        # Direction arrow
        dx = 2 * math.cos(state.theta)
        dy = 2 * math.sin(state.theta)
        ax.arrow(state.x, state.y, dx, dy, head_width=0.5, color=color, label=label)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Dubins Path: {path.path_type.name}')

    return ax
