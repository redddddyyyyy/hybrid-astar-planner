# Hybrid A* Path Planning

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?style=flat&logo=ros&logoColor=white)](https://docs.ros.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Implementation of the **Hybrid A\*** path planning algorithm for non-holonomic vehicles with **ROS2 integration**, **dynamic obstacle replanning**, **Dubins curves**, and **benchmark comparisons**.

---

## Results

### Path Planning Comparison

<p align="center">
  <img src="assets/benchmark_comparison.png" alt="Algorithm Comparison" width="900"/>
</p>

*Comparison of A*, RRT*, and Hybrid A* on parking lot scenario*

### Parking Lot Navigation

<p align="center">
  <img src="assets/parking_result.png" alt="Parking Scenario" width="700"/>
</p>

*Vehicle navigating through a parking lot with parked cars as obstacles*

### Path Planning Animation

<p align="center">
  <img src="assets/planning_demo.gif" alt="Planning Animation" width="700"/>
</p>

*Real-time visualization of vehicle following the planned path*

---

## Features

| Feature | Description |
|---------|-------------|
| **Hybrid A\*** | Grid-based search with continuous state tracking |
| **ROS2 Integration** | Full ROS2 package with nav_msgs/Path publishing |
| **Dynamic Replanning** | Real-time path updates when obstacles move |
| **Dubins Curves** | Optimal paths for forward-only motion |
| **Benchmark Suite** | Compare A*, RRT*, Hybrid A* performance |
| **Multiple Scenarios** | Parking, maze, narrow passage, U-turn demos |
| **Animated Visualization** | GIF export for path following |

---

## Project Structure

```
├── src/
│   ├── __init__.py          # Package exports
│   ├── hybrid_astar.py      # Main Hybrid A* algorithm
│   ├── vehicle.py           # Bicycle kinematic model
│   ├── grid.py              # Occupancy grid + collision detection
│   ├── visualization.py     # Plotting + animation
│   ├── dubins.py            # Dubins curve computation
│   ├── dynamic_planner.py   # Real-time replanning
│   └── benchmark.py         # A*, RRT*, Hybrid A* comparison
├── ros2_ws/                 # ROS2 workspace
│   └── src/hybrid_astar_planner/
│       ├── planner_node.py      # Main ROS2 node
│       ├── obstacle_publisher.py # Dynamic obstacle publisher
│       ├── config/              # Parameter files
│       └── launch/              # Launch files
├── main.py                  # Demo script
├── benchmark_demo.py        # Benchmark comparison script
├── examples/                # Example scripts
└── requirements.txt         # Dependencies
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/redddddyyyyy/hybrid-astar-planner.git
cd hybrid-astar-planner
pip install -r requirements.txt
```

### Run Demo

```bash
# Parking lot scenario
python main.py --scenario parking

# Save visualization
python main.py --scenario parking --save assets/parking_result.png

# Create animation GIF
python main.py --scenario parking --animate --save_gif assets/planning_demo.gif
```

### Run Benchmark

```bash
# Compare A*, RRT*, and Hybrid A*
python benchmark_demo.py --scenario parking --trials 5

# Save comparison plot
python benchmark_demo.py --scenario parking --save assets/benchmark_comparison.png
```

---

## Algorithm Comparison

| Algorithm | Type | Kinematics | Optimality | Speed |
|-----------|------|------------|------------|-------|
| **A\*** | Grid-based | Holonomic | Optimal* | Fast |
| **RRT\*** | Sampling | Holonomic | Asymptotic | Medium |
| **Hybrid A\*** | Grid-based | Non-holonomic | Near-optimal | Medium |

*\*Optimal for discretized grid, ignores vehicle constraints*

### Benchmark Results (Parking Scenario)

| Metric | A* | RRT* | Hybrid A* |
|--------|----:|-----:|----------:|
| Path Length (m) | 45.2 | 52.8 | 48.6 |
| Computation (ms) | 12 | 89 | 156 |
| Nodes Expanded | 1,842 | 3,500 | 2,156 |
| Kinematically Feasible | No | No | **Yes** |

---

## ROS2 Integration

### Build Package

```bash
cd ros2_ws
colcon build --packages-select hybrid_astar_planner
source install/setup.bash
```

### Launch Planner

```bash
ros2 launch hybrid_astar_planner planner.launch.py
```

### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/goal_pose` | geometry_msgs/PoseStamped | Goal input |
| `/initialpose` | geometry_msgs/PoseWithCovarianceStamped | Start input |
| `/map` | nav_msgs/OccupancyGrid | Costmap input |
| `/planned_path` | nav_msgs/Path | Planned path output |
| `/path_markers` | visualization_msgs/MarkerArray | RViz visualization |
| `/dynamic_obstacles` | visualization_msgs/MarkerArray | Moving obstacles |

### Dynamic Obstacle Testing

```bash
# Terminal 1: Launch planner
ros2 launch hybrid_astar_planner planner.launch.py

# Terminal 2: Publish moving obstacles
ros2 run hybrid_astar_planner obstacle_publisher

# Terminal 3: Set goal in RViz or via command
ros2 topic pub /goal_pose geometry_msgs/PoseStamped "{...}"
```

---

## Dubins Curves

Dubins paths are the shortest paths for vehicles that can only move forward with a minimum turning radius.

```python
from src import DubinsPlanner, State

planner = DubinsPlanner(turn_radius=5.0)
start = State(0, 0, 0)
goal = State(20, 10, 1.57)

path = planner.connect(start, goal)
```

### Path Types

| Type | Description |
|------|-------------|
| LSL | Left turn → Straight → Left turn |
| LSR | Left turn → Straight → Right turn |
| RSL | Right turn → Straight → Left turn |
| RSR | Right turn → Straight → Right turn |
| RLR | Right → Left → Right (no straight) |
| LRL | Left → Right → Left (no straight) |

---

## Dynamic Replanning

Handle moving obstacles with real-time path updates:

```python
from src import DynamicPlanner, DynamicObstacle, State
from src.grid import OccupancyGrid, GridConfig

# Setup
grid = OccupancyGrid(GridConfig(width=50, height=50))
planner = DynamicPlanner(grid)

# Add dynamic obstacle
obstacle = DynamicObstacle(x=25, y=25, radius=2.0, vx=0.5, vy=0.0, id=1)
planner.add_obstacle(obstacle)

# Plan initial path
start = State(5, 5, 0)
goal = State(45, 45, 0)
path = planner.plan_initial(start, goal)

# Start monitoring (triggers replan when path blocked)
planner.on_replan_triggered = lambda reason: print(f"Replanning: {reason}")
planner.start_monitoring()

# Update obstacle positions over time
planner.update_obstacles(dt=0.1)
```

---

## API Reference

### HybridAStar

```python
from src import HybridAStar, Vehicle, OccupancyGrid, State
from src.grid import GridConfig

grid = OccupancyGrid(GridConfig(width=50, height=50))
vehicle = Vehicle()
planner = HybridAStar(grid, vehicle)

path = planner.plan(
    start=State(5, 5, 0),
    goal=State(45, 45, 0)
)
```

### Benchmark

```python
from src import Benchmark

benchmark = Benchmark(grid, vehicle)
results = benchmark.run_comparison(
    start=(5, 5, 0),
    goal=(45, 45, 0)
)
benchmark.print_comparison(results)
```

---

## Vehicle Model

Uses the **bicycle kinematic model**:

```
                 Front wheel
                     ○
                     │
                     │  L (wheelbase)
                     │
        ────────────○────────────
              Rear wheel (reference point)
```

**Kinematics:**
```
ẋ = v · cos(θ)
ẏ = v · sin(θ)
θ̇ = v · tan(δ) / L
```

---

## References

- Dolgov, D., et al. "Path Planning for Autonomous Vehicles in Unknown Semi-structured Environments." *IJRR*, 2010.
- Dubins, L.E. "On Curves of Minimal Length with a Constraint on Average Curvature." *American Journal of Mathematics*, 1957.
- LaValle, S. M. *Planning Algorithms*. Cambridge University Press, 2006.
- Stanford Racing Team. "Junior: The Stanford Entry in the Urban Challenge." *JFR*, 2008.

---

## License

MIT
