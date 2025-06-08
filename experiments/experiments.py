import numpy as np
import os
import json
import matplotlib.pyplot as plt

from evopy.strategy import Strategy
from evopy.constraint_handling import ConstraintHandling
from evopy.initializers import InitializationStrategy
from evopy.results_manager import ResultsManager
from evopy.recombinations import RecombinationStrategy
from evopy.utils.combined_elbow import plot_combined_elbow
from problem_runner import CirclesInASquare


def create_defaults(
    n_circles: int,
    population_size: int,
    num_children: int,
    generations: int,
    results_manager: ResultsManager,
):
    """
    Creates default parameters for the evolutionary algorithm.
    """
    circles_defaults = {
        "n_circles": n_circles,
        "init_strategy": InitializationStrategy.RANDOM,
        "print_sols": False,
        "plot_sols": False,
        # "output_statistics": False,
        "init_jitter": 0.1,
        "results_manager": results_manager,
        "random_seed": None,
    }
    evolution_defaults = {
        "population_size": population_size,
        "num_children": num_children,
        "generations": generations,
        "strategy": Strategy.CMA,
        "constraint_handling": ConstraintHandling.RANDOM_REPAIR,
        "max_evaluations": 1e5,
        "max_run_time": None,
        "recombination_strategy": RecombinationStrategy.NONE,
        "elitism": False,
    }
    return circles_defaults, evolution_defaults


def run_experiments(
    options,
    random_seeds,
    param_to_overwrite,
    param_in_runner,
    circles_defaults,
    evolution_defaults,
    with_elitism=False,
):
    results = {}

    for seed in random_seeds:
        print(f"\nRunning with seed {seed}")
        for opt in options:
            elitism_vals = [True, False] if with_elitism else [False]
            for elitism in elitism_vals:
                key = f"{opt},elitism" if with_elitism and elitism else str(opt)
                print(
                    f"Testing {param_to_overwrite} = {opt}, elitism = {elitism}"
                    if with_elitism
                    else f"Testing {param_to_overwrite} = {opt}"
                )
                circles_args = {**circles_defaults, "random_seed": int(seed)}
                evolution_args = {**evolution_defaults, "elitism": elitism}

                if param_in_runner:
                    evolution_args[param_to_overwrite] = opt
                else:
                    circles_args[param_to_overwrite] = opt

                runner = CirclesInASquare(**circles_args)
                best_solution = runner.run_evolution_strategies(**evolution_args)

                results.setdefault(key, []).append(
                    {
                        "seed": int(seed),
                        "best_solution": (
                            best_solution.tolist()
                            if isinstance(best_solution, np.ndarray)
                            else best_solution
                        ),
                        "best_fitness": (
                            runner.best_total_score[-1]
                            if runner.best_total_score
                            else None
                        ),
                        "target_value": runner.get_target(),
                        "generations_run": len(runner.best_total_score),
                        "progression": runner.best_total_score,
                    }
                )
    return results


def analyze_and_plot_results(
    results, options, result_dir, title, param_to_overwrite, with_elitism=False
):
    os.makedirs(result_dir, exist_ok=True)

    analysis = {"seeds": [r["seed"] for r in next(iter(results.values()))]}
    target = next(iter(results.values()))[0]["target_value"]

    for key, opt_results in results.items():
        fitnesses = [r["best_fitness"] for r in opt_results]
        analysis[key] = {
            "mean_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses),
            "best_fitness": np.max(fitnesses),
            "worst_fitness": np.min(fitnesses),
            "target_value": target,
        }

    with open(os.path.join(result_dir, "detailed_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    with open(os.path.join(result_dir, "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=4)

    # Boxplot
    plt.figure(figsize=(12, 6))
    data = [[r["best_fitness"] for r in results[key]] for key in results]
    labels = [str(key) for key in results]
    plt.boxplot(data, tick_labels=labels)
    plt.axhline(y=target, color="r", linestyle="--", label="Target Value")
    plt.title(f"{title.capitalize()} Comparison")
    plt.ylabel("Best Fitness Achieved")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, f"{param_to_overwrite}_comparison.png"))
    plt.close()

    # Elbow plot
    plot_combined_elbow(
        results,
        title,
        list(results.keys()),
        os.path.join(result_dir, f"{param_to_overwrite}_progression.png"),
    )

    return analysis


def print_summary(analysis):
    """
    Print a summary of the analysis result
    @param analysis: analysis results
    """
    print()
    print("Experiment Results Summary:")
    print("==========================")
    for init, stats in analysis.items():
        if init == "seeds":
            continue
        print(f"\nScheme: {init}")
        print(f"Mean Fitness: {stats['mean_fitness']:.6f} ± {stats['std_fitness']:.6f}")
        print(f"Best Fitness: {stats['best_fitness']:.6f}")


def run_comparison(
    title: str,
    options: list | np.ndarray,
    param_to_overwrite: str,
    param_in_runner: bool = True,
    n_circles: int = 10,
    n_runs: int = 5,
    population_size: int = 30,
    num_children: int = 1,
    generations: int = 1000,
    random_seeds: np.ndarray = None,
    with_elitism: bool = False,
):
    """
    Runs a comparison between a list of options. For each option, n_runs are performed each with num_generations.
    The results are then saved to the output directory.
    @param title: Name of the options to compare. For example: Initialization Scheme
    @param options: The different options to run the experiment
    @param param_to_overwrite: Name of the parameter that has to be replaced by each of the options
    @param param_in_runner: Whether the parameter is in the circles or evolution parameters
    @param n_circles: Number of circles to run
    @param n_runs: Number of runs to run for each option
    @param population_size: Size of the population for each option and run
    @param num_children: Number of children for each option and run
    @param generations: Number of generations for each option and run
    @param random_seeds: An array of random seeds
    @param with_elitism: Whether to run the experiments with elitism
    @return: The results and the analysis that was used to plot.
    """
    if random_seeds is None:
        random_seeds = np.random.randint(0, 1000000, size=n_runs)

    results_manager = ResultsManager(
        f"{param_to_overwrite}_comparison", save_files=False
    )
    circles_defaults, evolution_defaults = create_defaults(
        n_circles, population_size, num_children, generations, results_manager
    )

    results = run_experiments(
        options,
        random_seeds,
        param_to_overwrite,
        param_in_runner,
        circles_defaults,
        evolution_defaults,
        with_elitism=with_elitism,
    )

    analysis = analyze_and_plot_results(
        results,
        options,
        results_manager.run_dir,
        title,
        param_to_overwrite,
        with_elitism=with_elitism,
    )

    print_summary(analysis)
    return results
