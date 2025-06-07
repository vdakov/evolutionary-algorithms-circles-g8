from experiments import run_comparison
from evopy.constraint_handling import ConstraintHandling

if __name__ == "__main__":
    # Run experiment
    results, analysis = run_comparison(
        "Constraint Handling",
        options=[ConstraintHandling("RR"), ConstraintHandling("BR")],
        param_to_overwrite="constraint_handling",
        param_in_runner=True,
        n_circles=10,
        n_runs=5,
        population_size=30,
        num_children=1,
        generations=100,
    )
