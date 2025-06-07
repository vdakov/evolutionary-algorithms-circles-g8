from evopy.recombinations import RecombinationStrategy
from experiments import run_comparison

if __name__ == "__main__":
    # Run experiment
    # results, analysis = run_single_comparison(
    results, analysis = run_comparison(
        "Recombination Strategy",
        options=[r for r in RecombinationStrategy],
        param_to_overwrite="recombination_strategy",
        param_in_runner=True,
        n_circles=10,
        n_runs=5,
        population_size=30,
        num_children=1,
        generations=100,
        with_elitism=True,
    )
