import time

import numpy as np

from evopy.individual import Individual
from evopy.progress_report import ProgressReport
from evopy.recombinations import RecombinationStrategy
from evopy.strategy import Strategy
from evopy.constraint_handling import run_random_repair, run_constraint_domination, ConstraintHandling
from evopy.utils import random_with_seed


class EvoPy:
    """Main class of the EvoPy package."""

    def __init__(
        self,
        fitness_function,
        individual_length,
        initial_population,
        generations=1000,
        population_size=30,
        num_children=1,
        maximize=False,
        strategy=Strategy.SINGLE,
        constraint_handling_func=run_random_repair,
        random_seed=None,
        reporter=None,
        target_fitness_value=None,
        target_tolerance=1e-5,
        max_run_time=None,
        max_evaluations=None,
        bounds=None,
        recombination_strategy=None,
        elitism=True,
        cma_state=None,
    ):
        """Initializes an EvoPy instance.

        :param fitness_function: the fitness function on which the individuals are evaluated
        :param individual_length: the length of each individual
        :param initial_population: the individual to start from
        :param generations: the number of generations to execute
        :param population_size: the population size of each generation
        :param num_children: the number of children generated per parent individual
        :param maximize: whether the fitness function should be maximized or minimized
        :param strategy: the strategy used to generate offspring by individuals. For more
                         information, check the Strategy enum
        :param constraint_handling_func: the strategy used to resolve invalid individuals
        :param random_seed: the seed to use for the random number generator
        :param reporter: callback to be invoked at each generation with a ProgressReport as argument
        :param target_fitness_value: target fitness value for early stopping
        :param target_tolerance: tolerance to within target fitness value is to be acquired
        :param max_run_time: maximum time allowed to run in seconds
        :param max_evaluations: maximum allowed number of fitness function evaluations
        :param bounds: bounds for the sampling the parameters of individuals
        """
        self.fitness_function = fitness_function
        self.individual_length = individual_length
        self.initial_population = initial_population
        self.generations = generations
        self.population_size = population_size
        self.num_children = num_children
        self.maximize = maximize
        self.strategy = strategy
        self.constraint_handling_func = constraint_handling_func
        self.random_seed = random_seed
        self.random = random_with_seed(self.random_seed)
        self.reporter = reporter
        self.target_fitness_value = target_fitness_value
        self.target_tolerance = target_tolerance
        self.max_run_time = max_run_time
        self.max_evaluations = max_evaluations
        self.bounds = bounds
        self.evaluations = 0
        self.recombination_strategy = recombination_strategy
        self.elitism = elitism
        self.cma_state = cma_state

    def _check_early_stop(self, start_time, best):
        """Check whether the algorithm can stop early, based on time and fitness target.

        :param start_time: the starting time to compare against
        :param best: the current best individual
        :return: whether the algorithm should be terminated early
        """
        return (
            (
                self.max_run_time is not None
                and (time.time() - start_time) > self.max_run_time
            )
            or (
                self.target_fitness_value is not None
                and abs(best.fitness - self.target_fitness_value)
                < self.target_tolerance
            )
            or (
                self.max_evaluations is not None
                and self.evaluations >= self.max_evaluations
            )
        )

    def run(self):
        """Run the evolutionary strategy algorithm.

        :return: the best genotype found
        """
        if self.individual_length == 0:
            return None

        start_time = time.time()

        # Initialize population and evaluate fitness
        population = self._init_population()
        population_fitness = np.array(
            [ind.evaluate(self.fitness_function) for ind in population]
        )
        # Track best solution
        best_idx = (
            np.argmax(population_fitness)
            if self.maximize
            else np.argmin(population_fitness)
        )
        best_ever = population[best_idx].copy()
        # Main loop
        for generation in range(self.generations):
            children_args = ()
            if self.recombination_strategy != RecombinationStrategy.NONE.value:
                # Normalize fitness for selection weights
                if self.maximize:
                    weights = population_fitness - np.min(population_fitness)
                else:
                    weights = np.max(population_fitness) - population_fitness
                weights = (
                    weights / np.sum(weights)
                    if np.sum(weights) > 0
                    else np.ones_like(weights) / len(weights)
                )
                children_args = (weights, population, self.recombination_strategy)
            # Generate offspring
            children = []
            if self.strategy == Strategy.CMA:
                children = self.cma_state.reproduce_cma(
                    population,
                    self.population_size,
                    self.fitness_function,
                    self.constraint_handling_func,
                    self.bounds,
                )
            else:
                for parent in population:
                    for _ in range(self.num_children):
                        children.append(parent.reproduce(*children_args))
            # Add elite if enabled
            if self.elitism:
                children.append(best_ever.copy())
            # Evaluate children
            children_fitness = np.array(
                [child.evaluate(self.fitness_function) for child in children])
            # Selection

            violation_errors = np.array(
                [ve if ve is not None else 0 for ve in [child.violation_error for child in children]])
            fitness_scores = np.array([child.evaluate(self.fitness_function) for child in children])

            fitness_key = -fitness_scores if self.maximize else fitness_scores

            if self.constraint_handling_func == run_constraint_domination:
                sorted_indices = np.lexsort((fitness_key, violation_errors))
            else:
                sorted_indices = np.argsort(fitness_key)

            # Update population
            population = [children[i] for i in sorted_indices[: self.population_size]]
            population_fitness = children_fitness[
                sorted_indices[: self.population_size]
            ]
            if self.constraint_handling_func == run_constraint_domination:
                pass
                # print(fitness_scores[sorted_indices])

            # Update best ever
            if (self.maximize and population_fitness[0] > best_ever.fitness) or (
                not self.maximize and population_fitness[0] < best_ever.fitness
            ):
                best_ever = population[0].copy()

            self.evaluations += len(children)
            # Report progress
            if self.reporter is not None:
                mean = np.mean(population_fitness)
                std = np.std(population_fitness)
                self.reporter(
                    ProgressReport(
                        generation,
                        self.evaluations,
                        population[0].genotype,
                        population_fitness[0],
                        mean,
                        std,
                        best_ever,
                    )
                )

            if self._check_early_stop(start_time, best_ever):
                break

        return best_ever.genotype

    def _init_population(self):
        # Initialize the strategy parameters
        if self.strategy == Strategy.SINGLE:
            strategy_parameters = np.asarray(self.random.randn(1))
        elif self.strategy == Strategy.MULTIPLE:
            strategy_parameters = np.asarray(self.random.randn(self.individual_length))
        elif self.strategy == Strategy.FULL_VARIANCE:
            strategy_parameters = np.asarray(
                self.random.randn(
                    int((self.individual_length + 1) * self.individual_length / 2)
                )
            )
        elif self.strategy == Strategy.CMA:
            strategy_parameters = None  # CMA does not use individual params
        else:
            raise ValueError(
                "Provided strategy parameter was not an instance of Strategy"
            )
        # Initialize population parameters
        population_parameters = np.asarray(
            self.constraint_handling_func(self, self.initial_population)
        )
        return [
            Individual(
                # Initialize genotype within possible bounds
                parameters,
                # Set strategy parameters
                self.strategy,
                strategy_parameters,
                self.constraint_handling_func,
                # Set seed and bounds for reproduction
                random_seed=self.random_seed,
                bounds=self.bounds,
            )
            for parameters in population_parameters
        ]
