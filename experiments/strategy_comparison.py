from experiments import run_single_comparison
from evopy.strategy import Strategy

if __name__ == "__main__":
    # Run experiment
    results, analysis = run_single_comparison(
        "strategy",
        options=[s for s in Strategy],
        param_to_overwrite="strategy",
        n_circles=10,
        n_runs=5,
        population_size=30,
        num_children=1,
        generations=100,
    )
    # Print summary
    print("\nExperiment Results Summary:")
    print("==========================")
    for strategy, stats in analysis.items():
        if strategy == "seeds":
            continue
        print(f"\nStrategy: {strategy}")
        print(f"Mean Fitness: {stats['mean_fitness']:.6f} ± {stats['std_fitness']:.6f}")
        print(f"Best Fitness: {stats['best_fitness']:.6f}")
