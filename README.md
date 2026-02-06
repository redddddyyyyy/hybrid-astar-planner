# Hybrid A* Path Planning

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Implementation of the **Hybrid A\*** path planning algorithm for non-holonomic vehicles. Combines grid-based A* search with continuous state tracking to find kinematically feasible paths.

---

## Results

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

## Overview

**Hybrid A\*** bridges the gap between grid-based planners (fast but ignore kinematics) and sampling-based planners (kinematically feasible but slow). It's widely used in autonomous vehicles, including by the Stanford Racing Team (DARPA Grand Challenge winner).

### Key Features

- **Non-holonomic constraints** — Bicycle kinematic model for realistic car-like motion
- **Continuous state tracking** — Smooth paths despite discrete grid search
- **Motion primitives** — Forward and reverse driving with variable steering
- **Path smoothing** — Gradient descent optimization for smoother trajectories
- **Multiple scenarios** — Parking, maze, narrow passage, U-turn demos
- **Animated visualization** — GIF export for path following

---

## Algorithm

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID A* ALGORITHM                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Discretize state space (x, y, θ) into grid cells        │
│                                                             │
│  2. Generate motion primitives from vehicle kinematics      │
│     ┌──────────────────────────────────────────────┐        │
│     │  Steering angles: [-35°, -17°, 0°, 17°, 35°] │        │
│     │  Directions: Forward, Reverse                │        │
│     └──────────────────────────────────────────────┘        │
│                                                             │
│  3. A* search with continuous state expansion:              │
│     • Pop lowest f-cost node                                │
│     • Expand using motion primitives                        │
│     • Check collisions with vehicle footprint               │
│     • Track continuous (x, y, θ) alongside grid cell        │
│                                                             │
│  4. Heuristic: Euclidean distance + heading penalty         │
│                                                             │
│  5. Post-process: Gradient descent path smoothing           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Cost Function

```
g(n) = g(parent) + distance + λ₁|steering| + λ₂|Δsteering| + λ₃·reverse + λ₄·direction_change
```

| Weight | Description | Purpose |
|--------|-------------|---------|
| λ₁ | Steering cost | Prefer straight paths |
| λ₂ | Steering change | Smooth steering transitions |
| λ₃ | Reverse cost | Prefer forward motion |
| λ₄ | Direction change | Minimize gear shifts |

---

## Project Structure

```
├── src/
│   ├── __init__.py          # Package exports
│   ├── hybrid_astar.py      # Main algorithm + path smoothing
│   ├── vehicle.py           # Bicycle kinematic model
│   ├── grid.py              # Occupancy grid + collision checking
│   └── visualization.py     # Plotting + animation
├── main.py                  # Demo script
├── assets/                  # Result images and GIFs
├── examples/                # Additional example scripts
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

# Maze navigation
python main.py --scenario maze

# Narrow passage
python main.py --scenario narrow_passage

# U-turn maneuver
python main.py --scenario u_turn

# Custom scenario
python main.py --scenario custom
```

### Save Results

```bash
# Save as image
python main.py --scenario parking --save assets/parking_result.png

# Create animation GIF
python main.py --scenario parking --animate --save_gif assets/planning_demo.gif
```

---

## Vehicle Model

Uses the **bicycle kinematic model** where the vehicle is represented by two wheels on the centerline:

```
                 Front wheel
                     ○
                     │
                     │  L (wheelbase)
                     │
        ────────────○────────────
              Rear wheel (reference point)
```

**State:** `(x, y, θ)` — Position of rear axle and heading angle

**Control:** `(δ, v)` — Steering angle and velocity

**Kinematics:**
```
ẋ = v · cos(θ)
ẏ = v · sin(θ)
θ̇ = v · tan(δ) / L
```

### Configuration

```python
from src.vehicle import VehicleConfig, Vehicle

config = VehicleConfig(
    wheelbase=2.5,      # Distance between axles (m)
    width=1.8,          # Vehicle width (m)
    length=4.5,         # Vehicle length (m)
    max_steer=0.6       # Maximum steering angle (rad)
)

vehicle = Vehicle(config)
```

---

## API Usage

### Basic Planning

```python
from src import HybridAStar, Vehicle, OccupancyGrid, Visualizer
from src.vehicle import State
from src.grid import GridConfig

# Create environment
grid_config = GridConfig(width=50, height=50, resolution=0.5)
grid = OccupancyGrid(grid_config)
grid.add_boundary(thickness=1.0)
grid.add_rectangle(25, 25, 10, 10)  # Obstacle

# Create planner
vehicle = Vehicle()
planner = HybridAStar(grid, vehicle)

# Plan path
start = State(x=5, y=5, theta=0)
goal = State(x=45, y=45, theta=0)
path = planner.plan(start, goal)

# Visualize
viz = Visualizer(grid, vehicle)
fig = viz.plot_planning_result(start, goal, path=path)
```

### Custom Planner Config

```python
from src.hybrid_astar import PlannerConfig

config = PlannerConfig(
    xy_resolution=1.0,          # Grid cell size for state hashing
    theta_resolution=0.1,       # Heading discretization (rad)
    num_steer_angles=7,         # Steering angle samples
    step_size=2.0,              # Motion primitive length (m)
    include_reverse=True,       # Allow reverse motion
    steer_cost=1.0,             # Steering penalty
    reverse_cost=3.0,           # Reverse penalty
    goal_xy_tolerance=1.0,      # Goal position tolerance (m)
    goal_theta_tolerance=0.2    # Goal heading tolerance (rad)
)

planner = HybridAStar(grid, vehicle, config)
```

---

## Scenarios

| Scenario | Description | Challenge |
|----------|-------------|-----------|
| `parking` | Navigate through parking lot | Tight spaces between parked cars |
| `maze` | Find path through maze walls | Multiple dead ends |
| `narrow_passage` | Pass through narrow corridor | Precise maneuvering required |
| `u_turn` | Execute U-turn around obstacle | Direction reversal |
| `custom` | User-defined obstacles | Fully configurable |

---

## References

- Dolgov, D., et al. "Path Planning for Autonomous Vehicles in Unknown Semi-structured Environments." *IJRR*, 2010.
- LaValle, S. M. *Planning Algorithms*. Cambridge University Press, 2006.
- Stanford Racing Team. "Junior: The Stanford Entry in the Urban Challenge." *JFR*, 2008.

---

## License

MIT
