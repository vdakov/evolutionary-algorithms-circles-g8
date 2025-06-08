import numpy as np
import matplotlib.pyplot as plt
import math

# Utility to create reproducible RNG

def random_with_seed(seed):
    return np.random.default_rng(seed)

# Initialization functions ensuring points lie within bounds

def random_init(n_circles, bounds=(0, 1), jitter=0.05, random_state=None):
    """Uniform random initialization with small jitter"""
    rng = random_with_seed(random_state)
    span = bounds[1] - bounds[0]
    # Base uniform sample
    xs = rng.uniform(bounds[0], bounds[1], n_circles)
    ys = rng.uniform(bounds[0], bounds[1], n_circles)
    # Add jitter as a fraction of span, then clip to bounds
    xs += rng.uniform(-jitter * span, jitter * span, n_circles)
    ys += rng.uniform(-jitter * span, jitter * span, n_circles)
    xs = np.clip(xs, bounds[0], bounds[1])
    ys = np.clip(ys, bounds[0], bounds[1])
    return np.column_stack((xs, ys)).ravel()


def grid_init(n_circles, bounds=(0, 1), jitter=0.02, random_state=None):
    """Grid-based initialization spanning full bounds with jitter"""
    rng = random_with_seed(random_state)
    dim = math.ceil(math.sqrt(n_circles))
    xs = np.linspace(bounds[0], bounds[1], dim)
    ys = np.linspace(bounds[0], bounds[1], dim)
    pts = []
    for x in xs:
        for y in ys:
            if len(pts) >= n_circles:
                break
            # jitter within small proportion of cell size
            cell = (bounds[1] - bounds[0]) / max(dim - 1, 1)
            dx = rng.uniform(-jitter * cell, jitter * cell)
            dy = rng.uniform(-jitter * cell, jitter * cell)
            pts.append((np.clip(x + dx, bounds[0], bounds[1]),
                        np.clip(y + dy, bounds[0], bounds[1])))
    arr = np.array(pts)
    return arr.ravel()


def concentric_init(n_circles, bounds=(0, 1), jitter=0.1, random_state=None):
    rng = random_with_seed(random_state)
    center = (bounds[0] + bounds[1]) / 2
    max_r = (bounds[1] - bounds[0]) / 2
    area_per = (math.pi * max_r**2) / n_circles
    ring = math.sqrt(area_per / math.pi)
    pts = []
    r = ring
    while len(pts) < n_circles:
        circ = 2 * math.pi * r
        count = max(1, int(circ / ring))
        for i in range(count):
            if len(pts) >= n_circles:
                break
            a = 2 * math.pi * i / count
            dr = rng.uniform(-jitter * ring, jitter * ring)
            da = rng.uniform(-jitter, jitter)
            rr = np.clip(r + dr, 0, max_r)
            x = center + rr * math.cos(a + da)
            y = center + rr * math.sin(a + da)
            pts.append((x, y))
        r += ring
    arr = np.array(pts)
    # Clip to bounds
    arr = np.clip(arr, bounds[0], bounds[1])
    return arr.ravel()


def edge_init(n_circles, bounds=(0, 1), jitter=0.2, random_state=None):
    """Place points preferentially near edges, others uniform"""
    rng = random_with_seed(random_state)
    pts = []
    span = bounds[1] - bounds[0]
    for _ in range(n_circles):
        if rng.random() < 0.5:
            edge = rng.integers(4)
            if edge == 0:       # left
                x = bounds[0] + jitter * span
                y = rng.uniform(bounds[0], bounds[1])
            elif edge == 1:     # right
                x = bounds[1] - jitter * span
                y = rng.uniform(bounds[0], bounds[1])
            elif edge == 2:     # bottom
                x = rng.uniform(bounds[0], bounds[1])
                y = bounds[0] + jitter * span
            else:               # top
                x = rng.uniform(bounds[0], bounds[1])
                y = bounds[1] - jitter * span
        else:
            # uniform fill
            x = rng.uniform(bounds[0], bounds[1])
            y = rng.uniform(bounds[0], bounds[1])
        pts.append((x, y))
    arr = np.array(pts)
    return arr.ravel()


def spiral_init(n_circles, bounds=(0, 1), jitter=0.05, random_state=None):
    rng = random_with_seed(random_state)
    center = (bounds[0] + bounds[1]) / 2
    max_r = (bounds[1] - bounds[0]) / 2
    phi = (1 + math.sqrt(5)) / 2
    pts = []
    for i in range(n_circles):
        angle = 2 * math.pi * phi * i
        r = max_r * math.sqrt(i / n_circles)
        dr = rng.uniform(-jitter * max_r, jitter * max_r)
        da = rng.uniform(-jitter, jitter)
        rr = np.clip(r + dr, 0, max_r)
        x = center + rr * math.cos(angle + da)
        y = center + rr * math.sin(angle + da)
        pts.append((x, y))
    arr = np.array(pts)
    arr = np.clip(arr, bounds[0], bounds[1])
    return arr.ravel()

# Dispatch dictionary
init_funcs = {
    'random': random_init,
    'grid': grid_init,
    'concentric': concentric_init,
    'edge': edge_init,
    'spiral': spiral_init
}

# Visualization for n=10
if __name__ == '__main__':
    n = 20
    bounds = (0, 1)
    seed = 42
    jitter = 0

    fig, axes = plt.subplots(1, 5, figsize=(12, 8))
    axes = axes.ravel()
    for idx, (name, func) in enumerate(init_funcs.items()):
        pts = func(n, bounds=bounds, jitter=jitter, random_state=seed)
        xs, ys = pts[::2], pts[1::2]
        ax = axes[idx]
        ax.scatter(xs, ys)
        ax.set_title(name.capitalize())
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_aspect('equal', 'box')
    # Hide extra subplot if exists
    if len(init_funcs) < len(axes):
        axes[-1].axis('off')
    plt.tight_layout()
    plt.show()
