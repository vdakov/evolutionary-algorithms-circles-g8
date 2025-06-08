from evopy.strategy import Strategy
from experiments import run_comparison
from evopy.constraint_handling import ConstraintHandling

if __name__ == "__main__":
    # Run experiment
    results = run_comparison(
        "CMA vs. Baseline",
        options=[s for s in Strategy],
        param_to_overwrite="strategy",
        param_in_runner=True,
        n_circles=10,
        n_runs=5,
        population_size=30,
        num_children=1,
        generations=1000,
    )
