from experiments import run_single_comparison, run_single_comparison_with_elitism
from evopy.constraint_handling import ConstraintHandling

if __name__ == "__main__":
    # Run experiment
    # results, analysis = run_single_comparison(
    results, analysis = run_single_comparison_with_elitism(
        "Constraint Handling",
        options=[ConstraintHandling("RR"), ConstraintHandling("BR")],
        param_to_overwrite="constraint_handling",
        param_in_runner=True,
        n_circles=10,
        n_runs=5,
        population_size=30,
        num_children=1,
        generations=500,
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
