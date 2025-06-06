import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def plot_combined_elbow(results, option_name, options, save_path):
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
            label=f"{option.value} (Mean)",
            linewidth=2,
        )
        plt.plot(
            generations,
            best_run,
            color=colors[idx],
            linestyle="--",
            label=f"{option.value} (Best Run)",
            linewidth=1.5,
        )
        plt.fill_between(
            generations, min_progression, max_progression, color=colors[idx], alpha=0.2
        )
    # Scaffold
    plt.title(f"{option_name.capitalize()} Progression Comparison")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
