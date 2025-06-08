import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os


def plot_combined_elbow(results, title, options, save_path):
    """Create an enhanced elbow plot showing progression of all options.
    For each option:
    - Solid line shows mean performance across runs
    - Shaded region shows min-max range across runs
    - Dashed line shows best performing run
    """
    plt.figure(figsize=(12, 6))
    colors = [
        matplotlib.colormaps["tab10"](i) for i in range(len(options))
    ]  # Use distinct colors for each option
    for idx, option in enumerate(options):
        option_results = results[str(option)]
        # Get progression data for all runs
        all_progressions = []
        for run in option_results:
            # Extract progression from best_solution history
            progression = []
            current_best = float("-inf")
            for gen in range(run["generations_run"]):
                if gen < len(run.get("progression", [])):
                    score = run["progression"][gen]
                    current_best = max(current_best, score)
                progression.append(current_best)
            all_progressions.append(progression)
        # Convert to numpy array for easier manipulation
        all_progressions = np.array(all_progressions)
        # Calculate statistics
        mean_progression = np.mean(all_progressions, axis=0)
        min_progression = np.min(all_progressions, axis=0)
        max_progression = np.max(all_progressions, axis=0)
        # Find the best run (one with highest final value)
        best_run_idx = np.argmax([prog[-1] for prog in all_progressions])
        best_run = all_progressions[best_run_idx]
        # Plot
        generations = range(1, len(mean_progression) + 1)
        plt.plot(
            generations,
            mean_progression,
            color=colors[idx],
            label=f"{str(option)} (Mean)",
            linewidth=2,
        )
        plt.plot(
            generations,
            best_run,
            color=colors[idx],
            linestyle="--",
            label=f"{str(option)} (Best Run)",
            linewidth=1.5,
        )
        plt.fill_between(
            generations, min_progression, max_progression, color=colors[idx], alpha=0.2
        )
    # Scaffold
    plt.title(f"{title.capitalize()} Progression Comparison")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def create_combined_plot(results, dir, n_circles_range, line_pad):
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(
        np.linspace(0, 1, n_circles_range[1] - n_circles_range[0] + 1)
    )

    for idx, n in enumerate(range(n_circles_range[0], n_circles_range[1] + 1)):
        runs = results[str(n)]
        target = runs[0]["target_value"]

        progressions = [np.array(run["progression"]) for run in runs]
        max_len = max(len(p) for p in progressions)

        padded = np.array(
            [
                np.pad(p, (0, max_len - len(p)), constant_values=target)
                for p in progressions
            ]
        )

        mean_progression = np.mean(padded, axis=0)
        generations = np.arange(max_len)

        color = colors[idx]
        plt.plot(generations, mean_progression, color=color, label=f"{n} circles")
        plt.plot(
            np.arange(-line_pad, max_len),
            [target] * (max_len + line_pad),
            color=color,
            linestyle="--",
            alpha=0.5,
        )

    plt.xlabel("Generation")
    plt.ylabel("Minimum Distance Between Circles")
    plt.title(
        "Performance Across Different Numbers of Circles\n(Dashed Lines Show Target Values)"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.ylim(0)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "benchmark.png"), bbox_inches="tight")
    plt.close()
