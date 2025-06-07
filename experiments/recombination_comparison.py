from evopy.recombinations import RecombinationStrategy
from experiments import run_single_comparison, run_single_comparison_with_elitism
from evopy.initializers import InitializationStrategy

if __name__ == "__main__":
    # Run experiment
    # results, analysis = run_single_comparison(
    results, analysis = run_single_comparison_with_elitism(
        "Recombination Strategy",
        options=[r for r in RecombinationStrategy],
        param_to_overwrite="recombination_strategy",
        param_in_runner=True,
        n_circles=10,
        n_runs=5,
        population_size=30,
        num_children=1,
        generations=100,
    )
    # Print summary
    print("\nExperiment Results Summary:")
    print("==========================")
    for init, stats in analysis.items():
        if init == "seeds":
            continue
        print(f"\nInitialization Scheme: {init}")
        print(f"Mean Fitness: {stats['mean_fitness']:.6f} ± {stats['std_fitness']:.6f}")
        print(f"Best Fitness: {stats['best_fitness']:.6f}")
