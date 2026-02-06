"""Simple example of Hybrid A* path planning."""

import sys
sys.path.insert(0, '..')

from src import HybridAStar, Vehicle, OccupancyGrid, Visualizer
from src.vehicle import State, VehicleConfig
from src.grid import GridConfig
from src.hybrid_astar import smooth_path
import matplotlib.pyplot as plt


def main():
    print("Simple Hybrid A* Planning Example")
    print("=" * 40)

    # 1. Create occupancy grid
    grid_config = GridConfig(
        width=40,   # 40 meters wide
        height=40,  # 40 meters tall
        resolution=0.25  # 25cm cells
    )
    grid = OccupancyGrid(grid_config)

    # Add boundary walls
    grid.add_boundary(thickness=0.5)

    # Add some obstacles
    grid.add_rectangle(x=20, y=15, width=8, height=4)   # Rectangle obstacle
    grid.add_rectangle(x=15, y=28, width=6, height=3)   # Another rectangle
    grid.add_circle(x=30, y=25, radius=3)               # Circular obstacle
    grid.add_circle(x=10, y=10, radius=2)               # Small circle

    print(f"Grid: {grid_config.width}m x {grid_config.height}m")
    print(f"Resolution: {grid_config.resolution}m/cell")
    print(f"Grid size: {grid_config.x_cells} x {grid_config.y_cells} cells")

    # 2. Create vehicle
    vehicle_config = VehicleConfig(
        wheelbase=2.5,   # 2.5m wheelbase
        width=1.8,       # 1.8m wide
        length=4.5,      # 4.5m long
        max_steer=0.6    # ~35 degrees max steering
    )
    vehicle = Vehicle(vehicle_config)

    print(f"\nVehicle: {vehicle_config.length}m x {vehicle_config.width}m")
    print(f"Min turn radius: {vehicle_config.min_turn_radius:.1f}m")

    # 3. Define start and goal
    start = State(x=5, y=5, theta=0.5)      # Bottom-left, facing NE
    goal = State(x=35, y=35, theta=0.5)     # Top-right, facing NE

    print(f"\nStart: ({start.x}, {start.y}), θ={start.theta:.2f} rad")
    print(f"Goal:  ({goal.x}, {goal.y}), θ={goal.theta:.2f} rad")

    # 4. Create planner and find path
    planner = HybridAStar(grid, vehicle)

    print("\nPlanning...")
    path = planner.plan(start, goal)

    if path is None:
        print("No path found!")
        return

    print(f"Found path with {len(path)} waypoints")

    # 5. Smooth the path
    smoothed = smooth_path(path, iterations=100)
    print(f"Smoothed to {len(smoothed)} waypoints")

    # 6. Visualize
    viz = Visualizer(grid, vehicle)
    fig = viz.plot_planning_result(
        start, goal,
        path=path,
        smoothed_path=smoothed,
        title="Hybrid A* - Simple Example"
    )

    plt.show()


if __name__ == "__main__":
    main()
