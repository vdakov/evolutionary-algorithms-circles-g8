"""THIS IS THE TEMPLATE FOR COMPARISON EXPERIMENTS
PARTS THAT ARE TO BE CHANGED ARE SURROUNDED WITH ##########################################
ALSO SOME PARTS ARE RENAMED TO 'opt' TO INDICATE THE TYPE OF OPTION TO BE TESTED"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from evopy.constraint_handling import ConstraintHandling
from evopy.initializers import InitializationStrategy
from evopy.results_manager import ResultsManager
from evopy.utils.combined_elbow import plot_combined_elbow
from main import CirclesInASquare
import json
import matplotlib.pyplot as plt


def run_strategy_comparison(
    n_circles=10,
    n_runs=5,
    population_size=30,
    num_children=1,
    generations=1000,
    random_seeds=None,
):
    ##########################################
    # Setup
    options = []  # ...
    ##########################################
    if random_seeds is None:
        random_seeds = np.random.randint(0, 1000000, size=n_runs)
    # Create results manager for the experiment
    ##########################################
    results_manager = ResultsManager("opt_comparison", save_files=False)
    ##########################################
    # Results storage
    results = {str(opt): [] for opt in options}
    # Run experiment
    for seed in random_seeds:
        print(f"\nRunning with seed {seed}")
        for opt in options:
            ##########################################
            print(f"Testing option: {opt.value}")
            ##########################################
            # Initialize runner
            runner = CirclesInASquare(
                n_circles=n_circles,
                print_sols=False,
                plot_sols=False,
                init_strategy=InitializationStrategy.RANDOM,
                init_jitter=0.1,
                results_manager=results_manager,
            )
            best_solution = runner.run_evolution_strategies(
                population_size=population_size,
                num_children=num_children,
                generations=generations,
                ##########################################
                strategy=opt,
                ##########################################
                constraint_handling=ConstraintHandling.RANDOM_REPAIR,
                max_evaluations=1e5,
                max_run_time=None,
                recombination_strategy=None,
                elitism=True,
                random_seed=int(seed),
            )
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
            "mean_gap_to_target": np.mean([abs(f - target) for f in fitnesses]),
            "best_gap_to_target": min([abs(f - target) for f in fitnesses]),
        }
    # Save detailed results
    results_path = os.path.join(results_manager.run_dir, "detailed_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    # Save analysis
    analysis_path = os.path.join(results_manager.run_dir, "analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=4)
    # Create plots
    plt.figure(figsize=(12, 6))
    data = [[r["best_fitness"] for r in results[str(opt)]] for opt in opt_results]
    plt.boxplot(data, tick_labels=[s.value for s in opt_results])
    plt.axhline(y=target, color="r", linestyle="--", label="Target Value")
    ##########################################
    plt.title("Option Comparison")
    ##########################################
    plt.ylabel("Best Fitness Achieved")
    plt.legend()
    plt.grid(True)
    ##########################################
    plt.savefig(os.path.join(results_manager.run_dir, "opt_comparison.png"))
    ##########################################
    plt.close()
    # Enhanced elbow plot
    plot_combined_elbow(
        results,
        opt_results,
        ##########################################
        os.path.join(results_manager.run_dir, "opt_progression.png"),
        ##########################################
    )
    return results, analysis


if __name__ == "__main__":
    # Run experiment
    ##########################################
    results, analysis = run_strategy_comparison(
        n_circles=10, n_runs=1, population_size=30, num_children=1, generations=10
    )
    ##########################################
    # Print summary
    print("\nExperiment Results Summary:")
    print("==========================")
    for opt, stats in analysis.items():
        if opt == "seeds":
            continue
        ##########################################
        print(f"\nOption: {opt}")
        ##########################################
        print(f"Mean Fitness: {stats['mean_fitness']:.6f} ± {stats['std_fitness']:.6f}")
        print(f"Best Fitness: {stats['best_fitness']:.6f}")
        print(f"Mean Gap to Target: {stats['mean_gap_to_target']:.6f}")
        print(f"Best Gap to Target: {stats['best_gap_to_target']:.6f}")
