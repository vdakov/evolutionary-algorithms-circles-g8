import matplotlib

matplotlib.use("Qt5Agg")

import math
import argparse
from evopy import EvoPy
from evopy import ProgressReport
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np
from evopy.strategy import Strategy
from evopy.constraint_handling import *

###########################################################
#                                                         #
# EvoPy framework from https://github.com/evopy/evopy     #
# Read documentation on github for further information.   #
#                                                         #
# Adjustments by Renzo Scholman:                          #
#       Added evaluation counter (also in ProgressReport) #
#       Added max evaluation stopping criterion           #
#       Added random repair for solution                  #
#       Added target fitness value tolerance              #
#                                                         #
# Original license stored in LICENSE file                 #
#                                                         #
# Install required dependencies with:                     #
#       pip install -r requirements.dev.txt               #
#                                                         #
###########################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Circles in a Square (CiaS) Evolutionary Algorithm"
    )
    # Problem configuration
    parser.add_argument(
        "--n_circles", type=int, default=10, help="Number of circles to pack"
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
        type=Strategy.from_string,
        choices=["single", "multiple", "full", "cma"],
        default="single",
        help="Variance strategy (single/multiple/full/cma)",
    )
    parser.add_argument(
        "--constraint_handling",
        type=ConstraintHandling.from_string,
        choices=[ConstraintHandling.BOUNDARY_REPAIR, ConstraintHandling.CONSTRAINT_DOMINATION, ConstraintHandling.RANDOM_REPAIR],
        default="BR",
        help="How to deal with out-of-bounds individuals: "
        "Boundary Repair (BD), Constraint domination (CD), or Random repair (RR)",
    )
    # Visualization and output
    parser.add_argument(
        "--skip_plot_sols", action="store_true", help="Plot solutions during evolution"
    )
    parser.add_argument(
        "--skip_print_sols",
        action="store_true",
        help="Print solutions during evolution",
    )
    # Early stopping criteria
    parser.add_argument(
        "--max_evals", type=int, default=1e5, help="Maximum number of evaluations"
    )
    parser.add_argument("--max_time", type=float, help="Maximum runtime in seconds")
    
    parser.add_argument("--elitism", type=bool, help="Elitism", default=False)
    parser.add_argument("--recombination_strategy",type=str,
        choices=["weighted","intermediate"],
        default=None,
        help="Recombination strategy to use",
    )
    # Parse and verify
    args = parser.parse_args()
    if not 2 <= args.n_circles:
        parser.error("Number of circles must be at least 20")
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
        self, n_circles, output_statistics=True, plot_sols=False, print_sols=False
    ):
        self.print_sols = print_sols
        self.output_statistics = output_statistics
        self.plot_best_sol = plot_sols
        self.n_circles = n_circles
        self.best_total_score = []
        self.fig = None
        self.ax = None

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
        values_to_reach = [
            1.414213562373095048801688724220,  # 2
            1.035276180410083049395595350499,  # 3
            1.000000000000000000000000000000,  # 4
            0.707106781186547524400844362106,  # 5
            0.600925212577331548853203544579,  # 6
            0.535898384862245412945107316990,  # 7
            0.517638090205041524697797675248,  # 8
            0.500000000000000000000000000000,  # 9
            0.421279543983903432768821760651,  # 10
            0.398207310236844165221512929748,
            0.388730126323020031391610191835,
            0.366096007696425085295389370603,
            0.348915260374018877918854409001,
            0.341081377402108877637121191351,
            0.333333333333333333333333333333,
            0.306153985300332915214516914060,
            0.300462606288665774426601772290,
            0.289541991994981660261698764510,
            0.286611652351681559449894454738,
        ]

        return values_to_reach[self.n_circles - 2]

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

        evopy = EvoPy(
            (
                circles_in_a_square
                if self.n_circles < 12
                else circles_in_a_square_scipy
            ),  # Fitness function
            self.n_circles * 2,  # Number of parameters
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
            elitism=elitism
        )

        best_solution = evopy.run()

        if self.plot_best_sol:
            plt.close()

        plt.ioff()
        plt.show()  # Keep the plot open at the end

        return best_solution


if __name__ == "__main__":
    args = parse_args()
    # Map string strategy to Strategy enum
    runner = CirclesInASquare(
        n_circles=args.n_circles,
        print_sols=not args.skip_print_sols,
        plot_sols=not args.skip_plot_sols,
    )
    constraint_func_dispatch = {
        ConstraintHandling.BOUNDARY_REPAIR: run_boundary_repair,
        ConstraintHandling.CONSTRAINT_DOMINATION: run_constraint_domination,
        ConstraintHandling.RANDOM_REPAIR: run_random_repair,
    }
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
