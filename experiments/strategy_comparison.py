from experiments import run_comparison
from evopy.strategy import Strategy

if __name__ == "__main__":
    # Run experiment
    results = run_comparison(
        "strategy",
        options=[s for s in Strategy],
        param_to_overwrite="strategy",
        n_circles=10,
        n_runs=5,
        population_size=30,
        num_children=1,
        generations=1000,
        with_elitism=False,
    )
