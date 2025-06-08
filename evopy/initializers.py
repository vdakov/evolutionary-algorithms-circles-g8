import numpy as np
from enum import Enum
from evopy.utils import random_with_seed
import math


class InitializationStrategy(str, Enum):
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
    """Grid-based initialization with spacing"""
    rng = random_with_seed(random_state)
    # Calculate grid dimensions
    grid_size = math.ceil(math.sqrt(n_circles))
    spacing = (bounds[1] - bounds[0]) / (grid_size + 1)
    points = []
    for x in np.linspace(bounds[0] + spacing, bounds[1] - spacing, grid_size):
        for y in np.linspace(bounds[0] + spacing, bounds[1] - spacing, grid_size):
            if len(points) // 2 >= n_circles:
                break
            # Add controlled jitter that ensures points stay within bounds
            max_jitter = min(
                spacing * jitter,
                min(x - bounds[0], bounds[1] - x, y - bounds[0], bounds[1] - y),
            )
            jitter_x = rng.uniform(-max_jitter, max_jitter)
            jitter_y = rng.uniform(-max_jitter, max_jitter)
            points.extend([x + jitter_x, y + jitter_y])
    return np.array(points[: n_circles * 2])


def concentric_init(n_circles, bounds=(0, 1), jitter=0.1, random_state=None):
    """Concentric circles initialization with spacing"""
    rng = random_with_seed(random_state)
    center = (bounds[1] + bounds[0]) / 2
    max_radius = (bounds[1] - bounds[0]) / 2
    # Calculate optimal number of rings based on area
    area_per_circle = (max_radius**2) / n_circles
    ring_spacing = math.sqrt(area_per_circle)
    points = []
    curr_radius = ring_spacing
    while len(points) < n_circles * 2:
        # Calculate number of points that fit on current ring
        circumference = 2 * math.pi * curr_radius
        n_points = max(1, int(circumference / ring_spacing))
        for i in range(n_points):
            if len(points) >= n_circles * 2:
                break
            angle = 2 * math.pi * i / n_points
            # Add controlled jitter
            r_jitter = rng.uniform(-jitter * ring_spacing, jitter * ring_spacing)
            angle_jitter = rng.uniform(
                -jitter * math.pi / n_points, jitter * math.pi / n_points
            )
            r = curr_radius + r_jitter
            angle_final = angle + angle_jitter
            x = center + r * math.cos(angle_final)
            y = center + r * math.sin(angle_final)
            points.extend([x, y])
        curr_radius += ring_spacing
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
