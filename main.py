import argparse
from evopy.strategy import Strategy
from evopy.constraint_handling import ConstraintHandling
from evopy.initializers import InitializationStrategy
from evopy.results_manager import ResultsManager
from evopy.recombinations import RecombinationStrategy
from problem_runner import CirclesInASquare

import matplotlib

matplotlib.use("Qt5Agg")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Circles in a Square (CiaS) Evolutionary Algorithm"
    )
    # Problem configuration
    parser.add_argument(
        "--n_circles", "-n", type=int, default=10, help="Number of circles to pack"
    )
    # Evolution Strategy parameters
    parser.add_argument(
        "--population_size", type=int, default=30, help="Population size"
    )
    parser.add_argument(
        "--remaining_population_factor_cma",
        type=int,
        default=2,
        help="A fraction of the population that will remain for the CMA strategy. 2 = half of the population",
    )
    parser.add_argument(
        "--num_children", type=int, default=1, help="Number of children per parent"
    )
    parser.add_argument(
        "--generations", type=int, default=1000, help="Maximum number of generations"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[s.value for s in Strategy],
        default="single",
        help="Variance strategy (single/multiple/full)",
    )
    # Extension parameters
    parser.add_argument(
        "--constraint_handling",
        type=str,
        choices=[s.value for s in ConstraintHandling],
        default="RR",
        help="How to deal with out-of-bounds individuals: "
        "Boundary Repair (BR), Constraint domination (CD), or Random repair (RR)",
    )
    parser.add_argument("--elitism", action="store_true", help="Elitism")
    parser.add_argument(
        "--recombination_strategy",
        type=str,
        choices=[r.value for r in RecombinationStrategy],
        default=RecombinationStrategy.NONE.value,
        help="Recombination strategy to use",
    )
    parser.add_argument(
        "--init_strategy",
        type=str,
        choices=[s.value for s in InitializationStrategy],
        default="random",
        help="Initialization strategy",
    )
    parser.add_argument(
        "--init_jitter",
        type=float,
        default=0.1,
        help="jitter/std amount for initialization strategies",
    )
    # Visualization and output
    parser.add_argument(
        "--skip_plot",
        action="store_true",
        help="Skip plotting solutions during evolution",
    )
    parser.add_argument(
        "--skip_print",
        action="store_true",
        help="Skip printing solutions during evolution",
    )
    # Early stopping criteria
    parser.add_argument(
        "--max_evals", type=int, default=1e5, help="Maximum number of evaluations"
    )
    parser.add_argument("--max_time", type=float, help="Maximum runtime in seconds")
    parser.add_argument(
        "--random_seed", type=int, help="Random seed for reproducibility"
    )
    # Parse, convert, and validate arguments
    parser_args = parser.parse_args()
    parser_args.strategy = Strategy.from_string(parser_args.strategy)
    parser_args.constraint_handling = ConstraintHandling.from_string(
        parser_args.constraint_handling
    )
    parser_args.recombination_strategy = RecombinationStrategy.from_string(
        parser_args.recombination_strategy
    )
    parser_args.init_strategy = InitializationStrategy.from_string(
        parser_args.init_strategy
    )
    return parser_args


if __name__ == "__main__":
    args = parse_args()
    # Initialize runner
    runner = CirclesInASquare(
        n_circles=args.n_circles,
        init_strategy=args.init_strategy,
        print_sols=not args.skip_print,
        plot_sols=not args.skip_plot,
        init_jitter=args.init_jitter,
        results_manager=ResultsManager(),
        remaining_population_factor_cma=args.remaining_population_factor_cma,
        random_seed=args.random_seed,
    )
    best = runner.run_evolution_strategies(
        population_size=args.population_size,
        num_children=args.num_children,
        generations=args.generations,
        strategy=args.strategy,
        constraint_handling=args.constraint_handling,
        max_evaluations=args.max_evals,
        recombination_strategy=args.recombination_strategy,
        elitism=args.elitism,
        max_run_time=args.max_time,
    )
