"""Demo script for Hybrid A* path planning."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src import HybridAStar, Vehicle, OccupancyGrid, Visualizer
from src.vehicle import VehicleConfig, State
from src.grid import GridConfig
from src.hybrid_astar import smooth_path, PlannerConfig
from src.visualization import create_scenario


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid A* Path Planning Demo")

    p.add_argument(
        "--scenario",
        type=str,
        default="parking",
        choices=["parking", "maze", "narrow_passage", "u_turn", "custom"],
        help="Predefined scenario to run"
    )
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save result to image file"
    )
    p.add_argument(
        "--animate",
        action="store_true",
        help="Create animation"
    )
    p.add_argument(
        "--save_gif",
        type=str,
        default=None,
        help="Save animation as GIF"
    )
    p.add_argument(
        "--no_smooth",
        action="store_true",
        help="Disable path smoothing"
    )

    return p.parse_args()


def run_custom_scenario() -> None:
    """Run a custom scenario with manual obstacle placement."""
    # Grid setup
    config = GridConfig(width=60, height=40, resolution=0.25)
    grid = OccupancyGrid(config)
    grid.add_boundary(thickness=0.5)

    # Add obstacles
    # Parking spots
    for i in range(4):
        grid.add_rectangle(15 + i * 8, 10, 5, 2.5)

    # Central obstacle
    grid.add_rectangle(30, 25, 10, 8)

    # Random circles
    grid.add_circle(50, 15, 3)
    grid.add_circle(10, 30, 2.5)

    # Vehicle
    vehicle_config = VehicleConfig(
        wheelbase=2.7,
        width=1.9,
        length=4.8,
        max_steer=0.6
    )
    vehicle = Vehicle(vehicle_config)

    # Planner
    planner_config = PlannerConfig(
        xy_resolution=1.0,
        theta_resolution=0.1,
        step_size=2.0,
        include_reverse=True
    )
    planner = HybridAStar(grid, vehicle, planner_config)

    # Start and goal
    start = State(5, 20, 0)
    goal = State(55, 20, 0)

    return grid, vehicle, planner, start, goal


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("HYBRID A* PATH PLANNING")
    print("=" * 60)
    print(f"Scenario: {args.scenario}")
    print()

    # Setup scenario
    if args.scenario == "custom":
        grid, vehicle, planner, start, goal = run_custom_scenario()
    else:
        grid, start, goal = create_scenario(args.scenario)
        vehicle = Vehicle()
        planner = HybridAStar(grid, vehicle)

    print(f"Start: ({start.x:.1f}, {start.y:.1f}, {start.theta:.2f} rad)")
    print(f"Goal:  ({goal.x:.1f}, {goal.y:.1f}, {goal.theta:.2f} rad)")
    print()

    # Plan path
    print("Planning...")
    path = planner.plan(start, goal)

    if path is None:
        print("Failed to find path!")
        return

    print(f"Path found with {len(path)} waypoints")

    # Smooth path
    smoothed = None
    if not args.no_smooth:
        print("Smoothing path...")
        smoothed = smooth_path(path, iterations=100)
        print(f"Smoothed path: {len(smoothed)} waypoints")

    # Visualize
    viz = Visualizer(grid, vehicle)

    if args.animate or args.save_gif:
        display_path = smoothed if smoothed else path
        anim = viz.create_animation(
            display_path, start, goal,
            interval=80,
            save_path=args.save_gif
        )
        if not args.save_gif:
            plt.show()
    else:
        fig = viz.plot_planning_result(
            start, goal,
            path=path,
            smoothed_path=smoothed,
            title=f"Hybrid A* - {args.scenario.replace('_', ' ').title()} Scenario"
        )

        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches='tight')
            print(f"Result saved to {args.save}")
        else:
            plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
