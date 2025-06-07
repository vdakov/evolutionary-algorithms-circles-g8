import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.metrics import euclidean_distances

from evopy import (
    ProgressReport,
    init_func_dipatch,
    EvoPy,
    constraint_func_dispatch,
    InitializationStrategy,
    ResultsManager,
    Strategy,
)
from evopy.optimal_values import optimal_values
from evopy.cma import CMAState


class CirclesInASquare:
    def __init__(
        self,
        n_circles,
        print_sols=True,
        plot_sols=True,
        output_statistics=True,
        init_strategy=None,
        init_jitter=0.1,
        results_manager=None,
        remaining_population_factor_cma=3,
        random_seed=None,
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
        self.results_manager = results_manager
        self.remaining_population_factor_cma = remaining_population_factor_cma
        self.random_seed = random_seed

        assert 2 <= n_circles <= 30

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

        # Update results manager if available
        if self.results_manager is not None:
            self.results_manager.update_progress(
                report.generation,
                report.best_fitness,
                report.avg_fitness,
                report.std_fitness,
                report.best_genotype,
            )

    def get_target(self):
        return optimal_values[self.n_circles - 2]

    def run_evolution_strategies(
        self,
        population_size,
        num_children,
        generations,
        strategy,
        constraint_handling,
        max_evaluations,
        max_run_time,
        recombination_strategy,
        elitism,
    ):
        remaining_population_cma = int(
            population_size / self.remaining_population_factor_cma
        )
        settings = {
            "n_circles": self.n_circles,
            "population_size": population_size,
            "num_children": num_children,
            "generations": generations,
            "strategy": strategy.value,
            "constraint_handling": constraint_handling.value,
            "max_evaluations": max_evaluations,
            "max_run_time": max_run_time,
            "recombination_strategy": recombination_strategy.value,
            "elitism": elitism,
            "init_strategy": self.init_strategy.value,
            "init_jitter": self.init_jitter,
            "remaining_population_cma": remaining_population_cma,
            "random_seed": self.random_seed,
        }
        self.results_manager.start_run(settings)

        callback = self.statistics_callback if self.output_statistics else None

        initial_population = np.asarray(
            [
                init_func_dipatch[self.init_strategy](
                    self.n_circles,
                    bounds=(0, 1),
                    jitter=self.init_jitter,
                    random_state=self.random_seed,
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
            constraint_handling_func=constraint_func_dispatch[constraint_handling],
            bounds=(0, 1),
            random_seed=self.random_seed,
            target_fitness_value=self.get_target(),
            max_evaluations=max_evaluations,
            max_run_time=max_run_time,
            recombination_strategy=recombination_strategy.value,
            elitism=elitism,
            cma_state=(
                CMAState(
                    self.n_circles * 2,
                    remaining_population_cma,
                    self.random_seed,
                )
                if strategy == Strategy.CMA
                else None
            ),
        )

        best_solution = evopy.run()

        if self.plot_best_sol:
            plt.close()

        plt.ioff()
        plt.show()  # Keep the plot open at the end

        # Save results
        if self.results_manager is not None:
            self.results_manager.save_results(best_solution, self.get_target())

        return best_solution


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
