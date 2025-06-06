import matplotlib

matplotlib.use("Qt5Agg")

import math
import argparse
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np

from evopy import EvoPy
from evopy import ProgressReport
from evopy.strategy import Strategy
from evopy.constraint_handling import *
from evopy.initializers import *
from evopy.optimal_values import optimal_values


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
        "Boundary Repair (BD), Constraint domination (CD), or Random repair (RR)",
    )
    parser.add_argument("--elitism", action="store_true", help="Elitism")
    parser.add_argument(
        "--recombination_strategy",
        type=str,
        choices=["weighted", "intermediate"],
        default=None,
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
    # Parse, convert, and validate arguments
    args = parser.parse_args()
    args.strategy = Strategy.from_string(args.strategy)
    args.constraint_handling = ConstraintHandling.from_string(args.constraint_handling)
    args.init_strategy = InitializationStrategy.from_string(args.init_strategy)
    if not 2 <= args.n_circles:
        parser.error("Number of circles must be at least 2")
    return args


# np/scipy CiaS implementation is faster for higher problem dimensions, i.e, more than 11 or 12 circles.
def circles_in_a_square_scipy(individual):
    points = np.reshape(individual, (-1, 2))
    dist = euclidean_distances(points)
    np.fill_diagonal(dist, 1e10)
    return np.min(dist)


# Pure python implementation is faster for lower problem dimensions
def circles_in_a_square(individual):
    n = len(individual)
    distances = []
    for i in range(0, n - 1, 2):
        for j in range(i + 2, n, 2):
            distances.append(
                math.sqrt(
                    math.pow((individual[i] - individual[j]), 2)
                    + math.pow((individual[i + 1] - individual[j + 1]), 2)
                )
            )
    return min(distances)


class CirclesInASquare:
    def __init__(
        self,
        n_circles,
        print_sols=True,
        plot_sols=True,
        output_statistics=True,
        init_strategy=None,
        init_jitter=0.1,
    ):
        self.print_sols = print_sols
        self.output_statistics = output_statistics
        self.plot_best_sol = plot_sols
        self.n_circles = n_circles
        self.best_total_score = []
        self.fig = None
        self.ax = None
        self.init_strategy = init_strategy
        self.init_jitter = init_jitter

        assert 2 <= n_circles <= 20

        if self.plot_best_sol:
            self.set_up_plot()

        if self.output_statistics:
            self.statistics_header()

    def set_up_plot(self):
        # Elbow plot
        plt.ion()
        self.elbow_fig, self.elbow_ax = plt.subplots()
        (self.elbow_line,) = self.elbow_ax.plot([], [], marker="o")
        self.elbow_ax.set_title("Best encountered score in any generation")
        self.elbow_ax.set_xlabel("Generation")
        self.elbow_ax.set_ylabel("Best Score so Far")
        self.elbow_ax.grid(True)
        # Circles plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("$x_0$")
        self.ax.set_ylabel("$x_1$")
        self.ax.set_title("Best solution in generation 0")
        self.fig.show()

    def statistics_header(self):
        if self.print_sols:
            print("Generation Evaluations Best-fitness (Best individual..)")
        else:
            print("Generation Evaluations Best-fitness")

    def statistics_callback(self, report: ProgressReport):
        output = "{:>10d} {:>11d} {:>12.8f} {:>12.8f} {:>12.8f}".format(
            report.generation,
            report.evaluations,
            report.best_fitness,
            report.avg_fitness,
            report.std_fitness,
        )

        if len(self.best_total_score) == 0:
            self.best_total_score.append(report.best_fitness)
        elif report.best_fitness <= self.best_total_score[-1]:
            self.best_total_score.append(self.best_total_score[-1])
        else:
            self.best_total_score.append(report.best_fitness)

        if self.print_sols:
            output += " ({:s})".format(np.array2string(report.best_genotype))
        print(output)

        if self.plot_best_sol:
            # Update elbow plot
            self.elbow_line.set_xdata(range(1, len(self.best_total_score) + 1))
            self.elbow_line.set_ydata(self.best_total_score)
            self.elbow_ax.relim()
            self.elbow_ax.autoscale_view()
            self.elbow_fig.canvas.draw()
            self.elbow_fig.canvas.flush_events()
            # Update circles plot
            points = np.reshape(report.best_genotype, (-1, 2))
            self.ax.clear()
            self.ax.scatter(points[:, 0], points[:, 1], clip_on=False, color="black")
            self.ax.set_xlim((0, 1))
            self.ax.set_ylim((0, 1))
            self.ax.set_title(
                "Best solution in generation {:d}".format(report.generation)
            )
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def get_target(self):
        return optimal_values[self.n_circles - 2]

    def run_evolution_strategies(
        self,
        population_size,
        num_children,
        generations,
        strategy,
        constraint_handling_func,
        max_evaluations,
        max_run_time,
        recombination_strategy,
        elitism,
    ):
        callback = self.statistics_callback if self.output_statistics else None

        initial_population = np.asarray(
            [
                init_func_dipatch[self.init_strategy](
                    self.n_circles,
                    bounds=(0, 1),
                    jitter=self.init_jitter,
                    random_state=None,
                )
                for _ in range(population_size)
            ]
        )

        evopy = EvoPy(
            (
                circles_in_a_square
                if self.n_circles < 12
                else circles_in_a_square_scipy
            ),  # Fitness function
            self.n_circles * 2,  # Number of parameters
            initial_population,
            reporter=callback,  # Prints statistics at each generation
            maximize=True,
            generations=generations,
            population_size=population_size,
            num_children=num_children,
            strategy=strategy,
            constraint_handling_func=constraint_handling_func,
            bounds=(0, 1),
            random_seed=42,
            target_fitness_value=self.get_target(),
            max_evaluations=max_evaluations,
            max_run_time=max_run_time,
            recombination_strategy=recombination_strategy,
            elitism=elitism,
        )

        best_solution = evopy.run()

        if self.plot_best_sol:
            plt.close()

        plt.ioff()
        plt.show()  # Keep the plot open at the end

        # TODO: Use reporter to print the best solution,
        # as well as plot the final elbow plot and best population solution
        return best_solution


if __name__ == "__main__":
    args = parse_args()
    runner = CirclesInASquare(
        n_circles=args.n_circles,
        print_sols=not args.skip_print,
        plot_sols=not args.skip_plot,
        init_strategy=args.init_strategy,
        init_jitter=args.init_jitter,
    )
    best = runner.run_evolution_strategies(
        population_size=args.population_size,
        num_children=args.num_children,
        generations=args.generations,
        strategy=args.strategy,
        constraint_handling_func=constraint_func_dispatch[args.constraint_handling],
        max_evaluations=args.max_evals,
        recombination_strategy=args.recombination_strategy,
        elitism=args.elitism,
        max_run_time=args.max_time,
    )
    # TODO: fetch dataset and compare with optimal solution
