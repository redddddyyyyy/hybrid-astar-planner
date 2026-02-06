"""Occupancy grid for obstacle representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class GridConfig:
    """Grid configuration parameters."""

    width: float = 50.0  # World width (m)
    height: float = 50.0  # World height (m)
    resolution: float = 0.5  # Cell size (m)

    def __post_init__(self):
        self.x_cells = int(self.width / self.resolution)
        self.y_cells = int(self.height / self.resolution)


class OccupancyGrid:
    """
    2D occupancy grid for collision checking.

    Stores binary obstacle map and precomputes distance transform
    for efficient collision checking and cost computation.
    """

    def __init__(self, config: GridConfig | None = None):
        self.config = config or GridConfig()

        # Binary occupancy grid (True = obstacle)
        self.grid = np.zeros(
            (self.config.y_cells, self.config.x_cells),
            dtype=bool
        )

        # Distance transform (distance to nearest obstacle)
        self._distance_map: np.ndarray | None = None

    @property
    def distance_map(self) -> np.ndarray:
        """Lazy-computed distance transform."""
        if self._distance_map is None:
            self._compute_distance_transform()
        return self._distance_map

    def _compute_distance_transform(self) -> None:
        """Compute Euclidean distance transform."""
        # Distance transform of free space
        self._distance_map = distance_transform_edt(~self.grid)
        self._distance_map *= self.config.resolution  # Convert to meters

    def _invalidate_distance_map(self) -> None:
        """Invalidate cached distance map after grid changes."""
        self._distance_map = None

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int(x / self.config.resolution)
        gy = int(y / self.config.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        x = (gx + 0.5) * self.config.resolution
        y = (gy + 0.5) * self.config.resolution
        return x, y

    def in_bounds(self, x: float, y: float) -> bool:
        """Check if world position is within grid bounds."""
        return (0 <= x < self.config.width and
                0 <= y < self.config.height)

    def is_occupied(self, x: float, y: float) -> bool:
        """Check if world position is occupied."""
        if not self.in_bounds(x, y):
            return True  # Out of bounds = occupied

        gx, gy = self.world_to_grid(x, y)
        return self.grid[gy, gx]

    def get_clearance(self, x: float, y: float) -> float:
        """Get distance to nearest obstacle at world position."""
        if not self.in_bounds(x, y):
            return 0.0

        gx, gy = self.world_to_grid(x, y)
        return self.distance_map[gy, gx]

    def check_collision_point(self, x: float, y: float, radius: float = 0.0) -> bool:
        """Check collision for a point with optional inflation."""
        return self.get_clearance(x, y) <= radius

    def check_collision_polygon(self, corners: np.ndarray) -> bool:
        """
        Check collision for a convex polygon.

        Uses point sampling along polygon edges for efficiency.

        Args:
            corners: Nx2 array of polygon vertices

        Returns:
            True if collision detected
        """
        n = len(corners)

        # Sample points along each edge
        for i in range(n):
            p1 = corners[i]
            p2 = corners[(i + 1) % n]

            # Number of samples based on edge length
            edge_len = np.linalg.norm(p2 - p1)
            num_samples = max(2, int(edge_len / self.config.resolution) + 1)

            for t in np.linspace(0, 1, num_samples):
                pt = p1 + t * (p2 - p1)
                if self.is_occupied(pt[0], pt[1]):
                    return True

        return False

    def add_rectangle(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        angle: float = 0.0
    ) -> None:
        """
        Add rectangular obstacle.

        Args:
            x, y: Center position
            width, height: Rectangle dimensions
            angle: Rotation angle (rad)
        """
        # Generate corners
        hw, hh = width / 2, height / 2
        corners_local = np.array([
            [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
        ])

        # Rotate
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        corners = corners_local @ R.T + np.array([x, y])

        # Rasterize polygon
        self._fill_polygon(corners)
        self._invalidate_distance_map()

    def add_circle(self, x: float, y: float, radius: float) -> None:
        """Add circular obstacle."""
        gx, gy = self.world_to_grid(x, y)
        r_cells = int(radius / self.config.resolution) + 1

        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                px, py = gx + dx, gy + dy
                if 0 <= px < self.config.x_cells and 0 <= py < self.config.y_cells:
                    wx, wy = self.grid_to_world(px, py)
                    if (wx - x) ** 2 + (wy - y) ** 2 <= radius ** 2:
                        self.grid[py, px] = True

        self._invalidate_distance_map()

    def add_boundary(self, thickness: float = 1.0) -> None:
        """Add boundary walls around the grid."""
        t_cells = max(1, int(thickness / self.config.resolution))

        # Top and bottom
        self.grid[:t_cells, :] = True
        self.grid[-t_cells:, :] = True

        # Left and right
        self.grid[:, :t_cells] = True
        self.grid[:, -t_cells:] = True

        self._invalidate_distance_map()

    def _fill_polygon(self, corners: np.ndarray) -> None:
        """Rasterize and fill a polygon."""
        from matplotlib.path import Path

        # Create path from corners
        path = Path(corners)

        # Check all grid cells
        for gy in range(self.config.y_cells):
            for gx in range(self.config.x_cells):
                wx, wy = self.grid_to_world(gx, gy)
                if path.contains_point((wx, wy)):
                    self.grid[gy, gx] = True

    def load_from_image(self, image_path: str, threshold: int = 128) -> None:
        """
        Load occupancy grid from image file.

        Dark pixels (< threshold) are treated as obstacles.
        """
        from PIL import Image

        img = Image.open(image_path).convert('L')
        img = img.resize((self.config.x_cells, self.config.y_cells))

        self.grid = np.array(img) < threshold
        self._invalidate_distance_map()

    def get_random_free_position(self, margin: float = 1.0) -> Tuple[float, float]:
        """Get a random position in free space."""
        import random

        max_attempts = 1000
        for _ in range(max_attempts):
            x = random.uniform(margin, self.config.width - margin)
            y = random.uniform(margin, self.config.height - margin)
            if self.get_clearance(x, y) > margin:
                return x, y

        raise RuntimeError("Could not find free position")
