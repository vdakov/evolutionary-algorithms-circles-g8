import numpy as np
from enum import Enum
from evopy.utils import random_with_seed
import math


class InitializationStrategy(Enum):
    RANDOM = "random"
    GRID = "grid"
    CONCENTRIC = "concentric"
    EDGE = "edge"
    SPIRAL = "spiral"

    @staticmethod
    def from_string(s: str):
        try:
            return InitializationStrategy(s)
        except ValueError:
            raise ValueError(f"Invalid initialization strategy: {s}")

    def __str__(self):
        return self.value


def random_init(n_circles, bounds=(0, 1), jitter=0.1, random_state=None):
    """Random initialization"""
    rng = random_with_seed(random_state)
    return rng.normal(0, jitter * 10, size=(n_circles * 2,))


def grid_init(n_circles, bounds=(0, 1), jitter=0.1, random_state=None):
    """Grid-based initialization with random jitter"""
    rng = random_with_seed(random_state)
    # Calculate grid dimensions
    grid_size = math.ceil(math.sqrt(n_circles))
    spacing = (bounds[1] - bounds[0]) / (grid_size + 1)
    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) // 2 >= n_circles:
                break
            # Add jitter
            x = (
                bounds[0]
                + spacing * (i + 1)
                + rng.uniform(-jitter * spacing, jitter * spacing)
            )
            y = (
                bounds[0]
                + spacing * (j + 1)
                + rng.uniform(-jitter * spacing, jitter * spacing)
            )
            points.extend([x, y])
    return np.array(points[: n_circles * 2])


def concentric_init(n_circles, bounds=(0, 1), jitter=0.1, random_state=None):
    """Concentric circles initialization with jitter"""
    rng = random_with_seed(random_state)
    center = (bounds[1] + bounds[0]) / 2
    max_radius = (bounds[1] - bounds[0]) / 2
    points = []
    circles_per_ring = [1]  # Center point
    radius_factor = 0.5
    while sum(circles_per_ring) < n_circles:
        # Each ring can fit more circles as radius increases
        next_ring = math.floor(2 * math.pi * len(circles_per_ring) / 2)
        circles_per_ring.append(next_ring)
    curr = 0
    # Iterate over each ring
    for ring_idx, n_points in enumerate(circles_per_ring):
        radius = max_radius * radius_factor * ring_idx
        for i in range(n_points):
            if curr >= n_circles:
                break
            angle = (2 * math.pi * i / n_points) + rng.uniform(-jitter, jitter)
            r = radius * (1 + rng.uniform(-jitter, jitter))
            # Add jitter
            x = center + r * math.cos(angle)
            y = center + r * math.sin(angle)
            points.extend([x, y])
            curr += 1
    return np.array(points[: n_circles * 2])


def edge_init(n_circles, bounds=(0, 1), jitter=0.1, random_state=None):
    """Edge-biased initialization"""
    rng = random_with_seed(random_state)
    points = []
    for _ in range(n_circles):
        if rng.random() < jitter:
            # Place point near an edge
            edge = rng.randint(4)  # Choose edge
            if edge < 2:  # Left or right edge
                x = bounds[edge]
                y = rng.uniform(bounds[0], bounds[1])
            else:  # Top or bottom edge
                x = rng.uniform(bounds[0], bounds[1])
                y = bounds[edge - 2]
            # Add small offset from edge
            offset = 0.05 * (bounds[1] - bounds[0])
            if edge == 0:  # Left edge
                x += offset
            elif edge == 1:  # Right edge
                x -= offset
            elif edge == 2:  # Bottom edge
                y += offset
            else:  # Top edge
                y -= offset
        else:
            # Random position
            x = rng.normal(0, jitter * 10)
            y = rng.normal(0, jitter * 10)
        points.extend([x, y])
    return np.array(points)


def spiral_init(n_circles, bounds=(0, 1), jitter=0.1, random_state=None):
    """Spiral-based initialization"""
    rng = random_with_seed(random_state)
    center = (bounds[1] + bounds[0]) / 2
    max_radius = (bounds[1] - bounds[0]) / 2
    points = []
    phi = (1 + math.sqrt(5)) / 2
    for i in range(n_circles):
        # Create spiral
        angle = 2 * math.pi * phi * i
        # Radius increases with each point
        radius = max_radius * math.sqrt(i / n_circles)
        # Add jitter to both angle and radius
        angle_jitter = rng.uniform(-jitter, jitter)
        radius_jitter = 1 + rng.uniform(-jitter, jitter)
        x = center + radius * radius_jitter * math.cos(angle + angle_jitter)
        y = center + radius * radius_jitter * math.sin(angle + angle_jitter)
        points.extend([x, y])
    return np.array(points)


init_func_dipatch = {
    InitializationStrategy.RANDOM: random_init,
    InitializationStrategy.GRID: grid_init,
    InitializationStrategy.CONCENTRIC: concentric_init,
    InitializationStrategy.EDGE: edge_init,
    InitializationStrategy.SPIRAL: spiral_init,
}
