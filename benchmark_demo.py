"""
Benchmark demo: Compare A*, RRT*, and Hybrid A* planners.

Generates comparison plots and statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.vehicle import Vehicle, VehicleConfig, State
from src.grid import OccupancyGrid, GridConfig
from src.benchmark import Benchmark, BenchmarkResult
from src.visualization import create_scenario


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark path planning algorithms")
    p.add_argument("--scenario", type=str, default="parking",
                   choices=["parking", "maze", "narrow_passage", "u_turn"])
    p.add_argument("--trials", type=int, default=3, help="Number of trials per scenario")
    p.add_argument("--save", type=str, default=None, help="Save comparison plot")
    return p.parse_args()


def plot_comparison(
    grid: OccupancyGrid,
    results: dict[str, BenchmarkResult],
    title: str = "Algorithm Comparison"
) -> plt.Figure:
    """Create comparison visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = {
        'A*': '#e74c3c',
        'RRT*': '#3498db',
        'Hybrid A*': '#27ae60'
    }

    # Plot paths on grid
    ax = axes[0]
    ax.imshow(
        grid.grid,
        origin='lower',
        extent=[0, grid.config.width, 0, grid.config.height],
        cmap='gray_r',
        alpha=0.8
    )

    for name, result in results.items():
        if result.success and result.path:
            x = [p[0] for p in result.path]
            y = [p[1] for p in result.path]
            ax.plot(x, y, color=colors[name], linewidth=2, label=name)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Planned Paths')
    ax.legend()
    ax.set_aspect('equal')

    # Bar chart: Path length
    ax = axes[1]
    names = []
    lengths = []
    for name, result in results.items():
        names.append(name)
        lengths.append(result.path_length if result.success else 0)

    bars = ax.bar(names, lengths, color=[colors[n] for n in names])
    ax.set_ylabel('Path Length (m)')
    ax.set_title('Path Length Comparison')

    for bar, length in zip(bars, lengths):
        if length > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{length:.1f}', ha='center', va='bottom')

    # Bar chart: Computation time
    ax = axes[2]
    times = [r.computation_time * 1000 for r in results.values()]  # Convert to ms

    bars = ax.bar(names, times, color=[colors[n] for n in names])
    ax.set_ylabel('Time (ms)')
    ax.set_title('Computation Time')

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{t:.1f}', ha='center', va='bottom')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def print_stats_table(stats: dict) -> None:
    """Print aggregated benchmark statistics."""
    print("\n" + "=" * 80)
    print("AGGREGATED BENCHMARK STATISTICS")
    print("=" * 80)
    print(f"{'Algorithm':<15} {'Success %':<12} {'Avg Length':<12} {'Avg Time':<12} {'Avg Nodes':<12} {'Smoothness':<12}")
    print("-" * 80)

    for name, s in stats.items():
        print(f"{name:<15} {s['success_rate']*100:<12.1f} {s['avg_length']:<12.2f} "
              f"{s['avg_time']*1000:<12.2f} {s['avg_nodes']:<12.0f} {s['avg_smoothness']:<12.4f}")

    print("=" * 80)


def main():
    args = parse_args()

    print("=" * 60)
    print("PATH PLANNING BENCHMARK")
    print("=" * 60)
    print(f"Scenario: {args.scenario}")
    print(f"Trials: {args.trials}")
    print()

    # Setup scenario
    grid, start, goal = create_scenario(args.scenario)
    vehicle = Vehicle()

    # Create benchmark
    benchmark = Benchmark(grid, vehicle)

    # Run single comparison for visualization
    print("Running single comparison for visualization...")
    results = benchmark.run_comparison(
        (start.x, start.y, start.theta),
        (goal.x, goal.y, goal.theta)
    )

    # Print results
    benchmark.print_comparison(results)

    # Run full benchmark suite
    print(f"\nRunning {args.trials} trials for statistics...")
    scenarios = [((start.x, start.y, start.theta), (goal.x, goal.y, goal.theta))]
    stats = benchmark.run_benchmark_suite(scenarios, num_trials=args.trials)

    print_stats_table(stats)

    # Create visualization
    fig = plot_comparison(grid, results, f"Algorithm Comparison - {args.scenario.title()}")

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
