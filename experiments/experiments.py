import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from evopy.strategy import Strategy
from evopy.constraint_handling import ConstraintHandling
from evopy.initializers import InitializationStrategy
from evopy.results_manager import ResultsManager
from evopy.utils.combined_elbow import plot_combined_elbow
from main import CirclesInASquare
import json
import matplotlib.pyplot as plt


def run_single_comparison(
    option_name,
    options,
    param_to_overwrite,
    param_in_runner=True,
    n_circles=10,
    n_runs=5,
    population_size=30,
    num_children=1,
    generations=1000,
    random_seeds=None,
):
    if random_seeds is None:
        random_seeds = np.random.randint(0, 1000000, size=n_runs)
    # Create results manager for the experiment
    results_manager = ResultsManager(
        f"{param_to_overwrite}_comparison", save_files=False
    )

    # Defaults for CirclesInASquare constructor
    circles_defaults = {
        "n_circles": n_circles,
        "print_sols": False,
        "plot_sols": False,
        "init_strategy": InitializationStrategy.RANDOM,
        "init_jitter": 0.1,
        "results_manager": results_manager,
        "random_seed": None,  # Will be set per run
    }
    # Defaults for run_evolution_strategies
    evolution_defaults = {
        "population_size": population_size,
        "num_children": num_children,
        "generations": generations,
        "strategy": Strategy.SINGLE,
        "constraint_handling": ConstraintHandling.RANDOM_REPAIR,
        "max_evaluations": 1e5,
        "max_run_time": None,
        "recombination_strategy": None,
        "elitism": False,
    }

    # Results storage
    results = {str(opt): [] for opt in options}
    # Run experiment
    for seed in random_seeds:
        print(f"\nRunning with seed {seed}")
        for opt in options:
            print(f"Testing {param_to_overwrite} = {opt}")
            # Construct arguments
            circles_args = {**circles_defaults, "random_seed": int(seed)}
            evolution_args = {
                **evolution_defaults,
            }
            # Tested parameter
            if param_in_runner:
                evolution_args[param_to_overwrite] = opt
            else:
                circles_args[param_to_overwrite] = opt
            # Initialize and run
            runner = CirclesInASquare(**circles_args)
            best_solution = runner.run_evolution_strategies(**evolution_args)
            # Store results
            results[str(opt)].append(
                {
                    "seed": int(seed),
                    "best_solution": (
                        best_solution.tolist()
                        if isinstance(best_solution, np.ndarray)
                        else best_solution
                    ),
                    "best_fitness": (
                        runner.best_total_score[-1] if runner.best_total_score else None
                    ),
                    "target_value": runner.get_target(),
                    "generations_run": len(runner.best_total_score),
                    "progression": runner.best_total_score,
                }
            )
    # Analyze results
    os.makedirs(results_manager.run_dir, exist_ok=True)
    analysis = {"seeds": random_seeds.tolist()}
    for opt in options:
        opt_results = results[str(opt)]
        fitnesses = [r["best_fitness"] for r in opt_results]
        target = opt_results[0]["target_value"]  # Same for all runs
        analysis[str(opt)] = {
            "mean_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses),
            "best_fitness": np.max(fitnesses),
            "worst_fitness": np.min(fitnesses),
            "target_value": target,
        }
    # Save detailed results
    results_path = os.path.join(results_manager.run_dir, "detailed_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    # Save analysis
    analysis_path = os.path.join(results_manager.run_dir, "analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=4)
    # Boxplot
    plt.figure(figsize=(12, 6))
    data = [[r["best_fitness"] for r in results[str(opt)]] for opt in options]
    plt.boxplot(data, tick_labels=[str(s) for s in options])
    plt.axhline(y=target, color="r", linestyle="--", label="Target Value")
    plt.title(f"{option_name.capitalize()} Comparison")
    plt.ylabel("Best Fitness Achieved")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(results_manager.run_dir, f"{param_to_overwrite}_comparison.png")
    )
    plt.close()
    # Enhanced elbow plot
    plot_combined_elbow(
        results,
        option_name,
        options,
        os.path.join(results_manager.run_dir, f"{param_to_overwrite}_progression.png"),
    )
    return results, analysis
