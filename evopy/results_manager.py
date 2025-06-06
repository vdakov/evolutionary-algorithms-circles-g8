import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class ResultsManager:
    """Class for managing and storing experiment results."""

    def __init__(self, experiment_name=None, save_files=True):
        """Initialize the results manager.

        Args:
            experiment_name: Optional name for the experiment. If None, uses timestamp.
        """
        self.experiment_name = experiment_name or "run"
        self.start_time = None
        self.settings = {}
        self.best_scores = []
        self.generation_scores = []
        self.save_files = save_files

    def start_run(self, settings):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(
            "outputs", f"{self.timestamp}_{self.experiment_name}"
        )
        # Create directory
        if self.save_files:
            os.makedirs(self.run_dir, exist_ok=True)
        # Init params
        self.start_time = time.time()
        self.settings = settings
        self.best_scores = []
        self.generation_scores = []

    def update_progress(
        self, generation, best_fitness, avg_fitness, std_fitness, best_genotype
    ):
        if len(self.best_scores) == 0 or best_fitness > self.best_scores[-1]:
            self.best_scores.append(best_fitness)
        else:
            self.best_scores.append(self.best_scores[-1])

        self.generation_scores.append(
            {
                "generation": generation,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "std_fitness": std_fitness,
                "best_genotype": (
                    best_genotype.tolist()
                    if isinstance(best_genotype, np.ndarray)
                    else best_genotype
                ),
            }
        )

    def save_results(self, final_solution, target_value):
        if not self.save_files:
            return
        # Save run settings
        settings_path = os.path.join(self.run_dir, "settings.json")
        with open(settings_path, "w") as f:
            json.dump(self.settings, f, indent=4)
        # Save progression data
        progress_path = os.path.join(self.run_dir, "progress.json")
        with open(progress_path, "w") as f:
            json.dump(self.generation_scores, f, indent=4)
        # Plot and save charts
        self._plot_elbow_chart(target_value)
        self._plot_final_solution(final_solution)
        # Save summary
        self._save_summary(final_solution, target_value)

    def _plot_elbow_chart(self, target_value):
        """Plot and save the elbow chart showing optimization progress."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_scores, label="Best Score")
        if target_value is not None:
            plt.axhline(y=target_value, color="r", linestyle="--", label="Target Value")
        plt.xlabel("Generation")
        plt.ylabel("Best Score")
        plt.ylim(bottom=0)
        plt.title("Optimization Progress")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.run_dir, "elbow_chart.png"))
        plt.close()

    def _plot_final_solution(self, solution):
        """Plot and save the final solution visualization."""
        points = np.reshape(solution, (-1, 2))
        plt.figure(figsize=(8, 8))
        plt.scatter(points[:, 0], points[:, 1], clip_on=False, color="black")
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.title("Final Solution")
        plt.savefig(os.path.join(self.run_dir, "final_solution.png"))
        plt.close()

    def _save_summary(self, final_solution, target_value):
        """Save a summary of the run results."""
        runtime = time.time() - self.start_time
        summary = {
            "runtime_seconds": runtime,
            "final_score": self.best_scores[-1] if self.best_scores else None,
            "target_value": target_value,
            "gap_to_target": (
                abs(self.best_scores[-1] - target_value)
                if self.best_scores and target_value
                else None
            ),
            "generations_run": len(self.generation_scores),
            "final_solution": (
                final_solution.tolist()
                if isinstance(final_solution, np.ndarray)
                else final_solution
            ),
        }
        summary_path = os.path.join(self.run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
