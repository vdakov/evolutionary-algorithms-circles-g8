import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from evopy import (
    Individual,
    init_func_dipatch,
    InitializationStrategy,
    run_boundary_repair,
    run_random_repair,
    run_constraint_domination,
    Strategy,
)


def draw_arrows(original_points, corrected_points, ax):
    for (x0, y0), (x1, y1) in zip(original_points, corrected_points):
        dx = x1 - x0
        dy = y1 - y0
        length = np.hypot(dx, dy)
        if length == 0:
            continue  # skip zero-length arrows

        shrink_ratio = 1  # Draw 90% of the way to the corrected point
        dx_shrink = dx * shrink_ratio
        dy_shrink = dy * shrink_ratio

        ax.arrow(
            x0,
            y0,
            dx_shrink,
            dy_shrink,
            head_width=0.05,
            head_length=0.1,
            fc="gray",
            ec="gray",
            alpha=0.7,
        )


def visualize_corrections(original_genotype, corrected_genotypes, titles):
    fig, axes = plt.subplots(1, len(corrected_genotypes), figsize=(15, 5))

    for idx, (corrected_genotype, ax) in enumerate(zip(corrected_genotypes, axes)):
        original_points = np.reshape(original_genotype, (-1, 2))
        corrected_points = np.reshape(corrected_genotype, (-1, 2))

        square = plt.Rectangle((0, 0), 1, 1, fill=False, color="black", lw=2)
        ax.add_patch(square)
        ax.set_title(titles[idx])

        if idx == 2 and individual is not None:  # Third graph: visualize penalty
            lower_violations = (
                np.maximum(individual.bounds[0] - original_genotype, 0) ** 2
            )
            upper_violations = (
                np.maximum(original_genotype - individual.bounds[1], 0) ** 2
            )
            total_violation = lower_violations + upper_violations

            # Sum every two values to get point-wise penalty (assuming 2D points)
            ax.scatter(
                original_points[:, 0],
                original_points[:, 1],
                color="blue",
                label="Original",
                alpha=0.6,
                edgecolors="black",
                s=200,
            )
            point_penalties = total_violation.reshape(-1, 2).sum(axis=1)

            draw_arrows(original_points, corrected_points, ax)
            scatter = ax.scatter(
                corrected_points[:, 0],
                corrected_points[:, 1],
                c=point_penalties,
                cmap="Reds",
                s=200,
                edgecolors="black",
            )

            fig.colorbar(scatter, ax=ax, label="Penalty")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        else:
            ax.scatter(
                original_points[:, 0],
                original_points[:, 1],
                color="blue",
                label="Original",
                alpha=0.6,
                edgecolors="black",
                s=200,
            )
            draw_arrows(original_points, corrected_points, ax)
            ax.scatter(
                corrected_points[:, 0],
                corrected_points[:, 1],
                color="red",
                label="Corrected",
                alpha=0.6,
                edgecolors="black",
                s=200,
            )
            ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from numpy.random import default_rng

    seed = 42
    rng = default_rng(seed)

    bounds = (0.0, 1.0)
    n_circles = 3
    dim = 2 * n_circles

    # Initialize genotype and strategy vectors
    genotype = rng.uniform(-0.5, 1.5, size=dim)  # intentionally some invalid
    strategy_parameters = {}

    # Create an Individual properly
    individual = Individual(
        genotype=genotype,
        strategy=Strategy.SINGLE,
        strategy_parameters=np.asarray([1]),
        constraint_handling_func=None,
        bounds=bounds,
        random_seed=seed,
    )

    # Apply repair methods
    repaired_boundary = run_boundary_repair(individual, np.copy(genotype))
    repaired_random = run_random_repair(individual, np.copy(genotype))
    constraint_domination = run_constraint_domination(individual, np.copy(genotype))

    # Plot the original and repaired individuals
    visualize_corrections(
        genotype,
        [repaired_boundary, repaired_random, constraint_domination],
        titles=["Boundary Repair", "Random Repair", "Constraint Domination"],
    )
