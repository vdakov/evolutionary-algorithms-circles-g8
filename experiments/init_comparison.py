from experiments import run_comparison
from evopy.initializers import InitializationStrategy

if __name__ == "__main__":
    # Run experiment
    results = run_comparison(
        "Initialization Scheme",
        options=[s for s in InitializationStrategy],
        param_to_overwrite="init_strategy",
        param_in_runner=False,
        n_circles=10,
        n_runs=5,
        population_size=30,
        num_children=1,
        generations=1000,
        with_elitism=True,
    )
