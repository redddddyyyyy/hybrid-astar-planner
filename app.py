"""
Streamlit web demo for Hybrid A* Path Planning.

Run with: streamlit run app.py
"""

import time
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

from src.vehicle import Vehicle, VehicleConfig, State
from src.grid import OccupancyGrid, GridConfig
from src.hybrid_astar import HybridAStar, PlannerConfig, smooth_path
from src.visualization import create_scenario


# Page config — sets the browser tab title and layout
st.set_page_config(
    page_title="Hybrid A* Path Planner",
    page_icon="🚗",
    layout="wide",
)

st.title("Hybrid A* Path Planner")
st.markdown(
    "Interactive demo of the **Hybrid A*** algorithm for non-holonomic vehicle "
    "path planning. Pick a scenario, tweak the settings, and hit **Plan Path**."
)

# ---------------------------------------------------------------------------
# Sidebar — all the controls live here so the main area stays clean
# ---------------------------------------------------------------------------

# SCENARIO PICKER
# WHY: Each scenario is a different obstacle layout (parking lot, maze, etc.)
# st.sidebar puts widgets in the left panel instead of the main page.
st.sidebar.header("Scenario")
scenario = st.sidebar.selectbox(
    "Choose a scenario",
    ["parking", "maze", "narrow_passage", "u_turn"],
    format_func=lambda s: s.replace("_", " ").title(),
)

# PLANNER SETTINGS
# WHY: These control how the A* search behaves. Letting users tweak them
# shows the tradeoffs (more steering angles = smoother but slower, etc.)
st.sidebar.header("Planner Settings")

num_steer = st.sidebar.slider(
    "Steering angles",
    min_value=3, max_value=9, value=5, step=2,
    help="Number of discrete steering angles to try at each step. "
         "More = smoother paths but slower search.",
)

step_size = st.sidebar.slider(
    "Step size (m)",
    min_value=0.5, max_value=3.0, value=1.5, step=0.25,
    help="How far the car moves in each expansion step. "
         "Smaller = more precise but slower.",
)

include_reverse = st.sidebar.checkbox(
    "Allow reverse",
    value=True,
    help="Let the car drive backward (needed for parking).",
)

smooth_enabled = st.sidebar.checkbox(
    "Smooth path",
    value=True,
    help="Apply gradient-descent smoothing to the raw path.",
)

# VEHICLE SETTINGS
# WHY: Different cars have different turning capabilities.
# A truck has a long wheelbase (wide turns), a go-kart has a short one (tight turns).
st.sidebar.header("Vehicle Settings")

wheelbase = st.sidebar.slider(
    "Wheelbase (m)",
    min_value=1.5, max_value=4.0, value=2.5, step=0.1,
    help="Distance between front and rear axles.",
)

max_steer_deg = st.sidebar.slider(
    "Max steering angle (deg)",
    min_value=15, max_value=60, value=35, step=5,
    help="Maximum angle the front wheels can turn.",
)


# ---------------------------------------------------------------------------
# Helper — draw the result with matplotlib
# ---------------------------------------------------------------------------
# WHY a separate function? We call it twice: once to preview the empty
# scenario, and once after planning to show the path. Putting it in a
# function avoids copy-pasting the same drawing code.

def create_figure(grid, vehicle, start, goal, path, smoothed_path):
    """Create a matplotlib figure showing the planning result."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # 1. Draw the obstacle grid (black = obstacle, white = free)
    ax.imshow(
        grid.grid,
        origin="lower",
        extent=[0, grid.config.width, 0, grid.config.height],
        cmap="gray_r",
        alpha=0.8,
    )

    # 2. Draw raw path (faint red line — the jagged original)
    if path:
        px = [s.x for s in path]
        py = [s.y for s in path]
        ax.plot(px, py, color="#e74c3c", linewidth=1.5, alpha=0.5, label="Raw Path")

    # 3. Draw smoothed path (bold green line — the clean result)
    if smoothed_path:
        sx = [s.x for s in smoothed_path]
        sy = [s.y for s in smoothed_path]
        ax.plot(sx, sy, color="#27ae60", linewidth=3, label="Smoothed Path")

        # Heading arrows so you can see which way the car faces
        step = max(1, len(smoothed_path) // 12)
        for i in range(0, len(smoothed_path), step):
            s = smoothed_path[i]
            dx = 0.6 * math.cos(s.theta)
            dy = 0.6 * math.sin(s.theta)
            ax.arrow(
                s.x, s.y, dx, dy,
                head_width=0.25, head_length=0.15,
                fc="#27ae60", ec="#27ae60", alpha=0.7,
            )

    # 4. Draw the car at start (blue) and goal (purple)
    for state, color, label in [(start, "#3498db", "Start"), (goal, "#9b59b6", "Goal")]:
        fp = vehicle.get_footprint(state)
        poly = patches.Polygon(
            fp, closed=True,
            facecolor=color, edgecolor="black",
            alpha=0.85, linewidth=1.5, label=label,
        )
        ax.add_patch(poly)
        # White arrow showing which direction the car faces
        arrow_len = vehicle.config.length * 0.4
        ax.arrow(
            state.x, state.y,
            arrow_len * math.cos(state.theta),
            arrow_len * math.sin(state.theta),
            head_width=0.3, head_length=0.2,
            fc="white", ec="black",
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title(
        f"Hybrid A* — {scenario.replace('_', ' ').title()}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main area — Plan button + results
# ---------------------------------------------------------------------------
# WHY two columns? Left column (wide) shows the plot, right column (narrow)
# shows a quick explanation. Keeps the layout clean.

col_left, col_right = st.columns([3, 1])

with col_right:
    st.subheader("How it works")
    st.markdown(
        """
        1. **Grid search** expands nodes using bicycle-model kinematics
        2. **Dubins shortcut** tries a smooth curve to the goal every 10 iterations
        3. **Path smoothing** pulls waypoints toward their neighbors

        The algorithm respects the car's **steering limits** and **minimum turning
        radius**, producing paths a real vehicle can follow.
        """
    )

with col_left:
    plan_button = st.button("Plan Path", type="primary", use_container_width=True)

    if plan_button:
        # ---- USER CLICKED "PLAN PATH" ----

        # Build the scenario (obstacles + start/goal positions)
        grid, start, goal = create_scenario(scenario)

        # Build the car with user's settings
        vehicle_config = VehicleConfig(
            wheelbase=wheelbase,
            max_steer=math.radians(max_steer_deg),
        )
        vehicle = Vehicle(vehicle_config)

        # Build the planner with user's settings
        planner_config = PlannerConfig(
            num_steer_angles=num_steer,
            step_size=step_size,
            include_reverse=include_reverse,
        )
        planner = HybridAStar(grid, vehicle, planner_config)

        # Run the planner and time it
        with st.spinner("Planning path..."):
            t0 = time.perf_counter()
            path = planner.plan(start, goal)
            elapsed = time.perf_counter() - t0

        if path is None:
            st.error("No path found! Try adjusting the settings.")
        else:
            # Smooth the path if enabled
            smoothed = None
            if smooth_enabled:
                smoothed = smooth_path(path, iterations=100)

            display_path = smoothed if smoothed else path

            # Calculate total path length
            path_len = sum(
                display_path[i].distance_to(display_path[i + 1])
                for i in range(len(display_path) - 1)
            )

            # Show stats as 4 metric cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Computation", f"{elapsed * 1000:.0f} ms")
            m2.metric("Waypoints", len(display_path))
            m3.metric("Path Length", f"{path_len:.1f} m")
            m4.metric("Smoothed", "Yes" if smoothed else "No")

            # Draw the result
            fig = create_figure(grid, vehicle, start, goal, path, smoothed)
            st.pyplot(fig)
            plt.close(fig)
    else:
        # ---- NO CLICK YET — show empty scenario preview ----
        grid, start, goal = create_scenario(scenario)
        vehicle = Vehicle(VehicleConfig(
            wheelbase=wheelbase,
            max_steer=math.radians(max_steer_deg),
        ))
        fig = create_figure(grid, vehicle, start, goal, None, None)
        st.pyplot(fig)
        plt.close(fig)
