from experiments import run_benchmark
from evopy.constraint_handling import ConstraintHandling
from evopy.strategy import Strategy
from evopy.initializers import InitializationStrategy

if __name__ == "__main__":
    # Run experiment
    results = run_benchmark(
        range(2, 21),
        Strategy.CMA,
        InitializationStrategy.GRID,
        ConstraintHandling.BOUNDARY_REPAIR,
        n_runs=5,
        population_size=50,
        num_children=1,
        generations=1000,
    )
