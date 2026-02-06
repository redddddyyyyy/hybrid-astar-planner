"""Visualization utilities for Hybrid A* planning."""

from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np

from .vehicle import Vehicle, State
from .grid import OccupancyGrid


class Visualizer:
    """Visualization tools for path planning."""

    # Color scheme
    COLORS = {
        'obstacle': '#2c3e50',
        'free': '#ecf0f1',
        'path': '#e74c3c',
        'smoothed_path': '#27ae60',
        'start': '#3498db',
        'goal': '#9b59b6',
        'vehicle': '#f39c12',
        'explored': '#bdc3c7',
    }

    def __init__(self, grid: OccupancyGrid, vehicle: Vehicle):
        self.grid = grid
        self.vehicle = vehicle

    def plot_environment(
        self,
        ax: plt.Axes | None = None,
        show_distance_field: bool = False
    ) -> plt.Axes:
        """Plot the environment grid."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        if show_distance_field:
            # Show distance transform
            im = ax.imshow(
                self.grid.distance_map,
                origin='lower',
                extent=[0, self.grid.config.width, 0, self.grid.config.height],
                cmap='Blues',
                alpha=0.8
            )
            plt.colorbar(im, ax=ax, label='Distance to obstacle (m)')
        else:
            # Show binary occupancy
            ax.imshow(
                self.grid.grid,
                origin='lower',
                extent=[0, self.grid.config.width, 0, self.grid.config.height],
                cmap='gray_r',
                alpha=0.8
            )

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_vehicle(
        self,
        state: State,
        ax: plt.Axes,
        color: str | None = None,
        alpha: float = 1.0,
        label: str | None = None
    ) -> None:
        """Plot vehicle at given state."""
        color = color or self.COLORS['vehicle']
        footprint = self.vehicle.get_footprint(state)

        # Vehicle body
        polygon = patches.Polygon(
            footprint,
            closed=True,
            facecolor=color,
            edgecolor='black',
            alpha=alpha,
            linewidth=1.5,
            label=label
        )
        ax.add_patch(polygon)

        # Direction arrow
        arrow_len = self.vehicle.config.length * 0.4
        dx = arrow_len * np.cos(state.theta)
        dy = arrow_len * np.sin(state.theta)
        ax.arrow(
            state.x, state.y, dx, dy,
            head_width=0.3,
            head_length=0.2,
            fc='white',
            ec='black',
            alpha=alpha
        )

    def plot_path(
        self,
        path: List[State],
        ax: plt.Axes,
        color: str | None = None,
        linewidth: float = 2.0,
        label: str | None = None,
        show_heading: bool = False,
        alpha: float = 1.0
    ) -> None:
        """Plot path as a line."""
        if not path:
            return

        color = color or self.COLORS['path']
        x = [s.x for s in path]
        y = [s.y for s in path]

        ax.plot(x, y, color=color, linewidth=linewidth, label=label, alpha=alpha)

        if show_heading:
            # Show heading arrows at intervals
            step = max(1, len(path) // 10)
            for i in range(0, len(path), step):
                s = path[i]
                dx = 0.5 * np.cos(s.theta)
                dy = 0.5 * np.sin(s.theta)
                ax.arrow(s.x, s.y, dx, dy, head_width=0.15, head_length=0.1,
                        fc=color, ec=color, alpha=0.7)

    def plot_start_goal(
        self,
        start: State,
        goal: State,
        ax: plt.Axes
    ) -> None:
        """Plot start and goal positions."""
        self.plot_vehicle(start, ax, color=self.COLORS['start'], label='Start')
        self.plot_vehicle(goal, ax, color=self.COLORS['goal'], label='Goal')

    def plot_planning_result(
        self,
        start: State,
        goal: State,
        path: List[State] | None = None,
        smoothed_path: List[State] | None = None,
        title: str = "Hybrid A* Path Planning"
    ) -> plt.Figure:
        """Create complete visualization of planning result."""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Environment
        self.plot_environment(ax)

        # Paths
        if path:
            self.plot_path(path, ax, color=self.COLORS['path'],
                          linewidth=2, label='Raw Path')

        if smoothed_path:
            self.plot_path(smoothed_path, ax, color=self.COLORS['smoothed_path'],
                          linewidth=3, label='Smoothed Path', show_heading=True)

        # Start and goal
        self.plot_start_goal(start, goal, ax)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')

        plt.tight_layout()
        return fig

    def create_animation(
        self,
        path: List[State],
        start: State,
        goal: State,
        interval: int = 100,
        save_path: str | None = None
    ) -> FuncAnimation:
        """
        Create animation of vehicle following path.

        Args:
            path: Path to animate
            start: Start state
            goal: Goal state
            interval: Frame interval in ms
            save_path: If provided, save animation to file

        Returns:
            Animation object
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        def init():
            ax.clear()
            self.plot_environment(ax)
            self.plot_path(path, ax, alpha=0.3)
            self.plot_vehicle(start, ax, color=self.COLORS['start'], alpha=0.3)
            self.plot_vehicle(goal, ax, color=self.COLORS['goal'], alpha=0.3)
            ax.set_title('Hybrid A* Path Planning', fontsize=14, fontweight='bold')
            return []

        def animate(frame):
            ax.clear()
            self.plot_environment(ax)

            # Path traveled so far
            if frame > 0:
                self.plot_path(path[:frame + 1], ax,
                              color=self.COLORS['smoothed_path'], linewidth=2)

            # Current vehicle position
            self.plot_vehicle(path[frame], ax, color=self.COLORS['vehicle'])

            # Start and goal (faded)
            self.plot_vehicle(start, ax, color=self.COLORS['start'], alpha=0.3)
            self.plot_vehicle(goal, ax, color=self.COLORS['goal'], alpha=0.3)

            ax.set_title(f'Hybrid A* Path Planning - Step {frame + 1}/{len(path)}',
                        fontsize=14, fontweight='bold')
            return []

        anim = FuncAnimation(
            fig, animate, init_func=init,
            frames=len(path), interval=interval, blit=False
        )

        if save_path:
            anim.save(save_path, writer='pillow', fps=1000 // interval)
            print(f"Animation saved to {save_path}")

        return anim


def create_scenario(
    scenario: str = "parking"
) -> Tuple[OccupancyGrid, State, State]:
    """
    Create predefined test scenarios.

    Args:
        scenario: One of "parking", "maze", "narrow_passage", "u_turn"

    Returns:
        (grid, start_state, goal_state)
    """
    from .grid import GridConfig

    if scenario == "parking":
        # Parking lot scenario
        config = GridConfig(width=40, height=30, resolution=0.25)
        grid = OccupancyGrid(config)
        grid.add_boundary(thickness=0.5)

        # Parked cars (rectangles)
        for i in range(5):
            grid.add_rectangle(8 + i * 6, 8, 4.5, 2.0)
            grid.add_rectangle(8 + i * 6, 22, 4.5, 2.0)

        start = State(5, 15, 0)
        goal = State(35, 15, 0)

    elif scenario == "maze":
        # Simple maze
        config = GridConfig(width=50, height=50, resolution=0.5)
        grid = OccupancyGrid(config)
        grid.add_boundary(thickness=1.0)

        # Walls
        grid.add_rectangle(15, 25, 2, 30)
        grid.add_rectangle(35, 25, 2, 30)
        grid.add_rectangle(25, 40, 18, 2)

        start = State(5, 5, np.pi / 4)
        goal = State(45, 45, np.pi / 4)

    elif scenario == "narrow_passage":
        # Narrow corridor
        config = GridConfig(width=50, height=20, resolution=0.25)
        grid = OccupancyGrid(config)
        grid.add_boundary(thickness=0.5)

        # Narrowing walls
        grid.add_rectangle(20, 5, 15, 3)
        grid.add_rectangle(20, 15, 15, 3)

        start = State(5, 10, 0)
        goal = State(45, 10, 0)

    elif scenario == "u_turn":
        # U-turn scenario
        config = GridConfig(width=30, height=40, resolution=0.25)
        grid = OccupancyGrid(config)
        grid.add_boundary(thickness=0.5)

        # Central obstacle forcing U-turn
        grid.add_rectangle(15, 20, 20, 15)

        start = State(5, 10, 0)
        goal = State(5, 30, np.pi)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return grid, start, goal
