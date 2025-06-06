import time

import numpy as np

from evopy.individual import Individual
from evopy.progress_report import ProgressReport
from evopy.strategy import Strategy
from evopy.constraint_handling import run_random_repair
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

        population = self._init_population()
        best = sorted(
            population,
            reverse=self.maximize,
            key=lambda individual: individual.evaluate(self.fitness_function),
        )[0].copy()

        for generation in range(self.generations):

            children_args = ()

            if self.recombination_strategy:
                fitnesses = [
                    individual.evaluate(self.fitness_function)
                    for individual in population
                ]
                total_fitness = sum(fitnesses)
                weights = np.divide(fitnesses, total_fitness)
                children_args = (weights, population, self.recombination_strategy)

            start_index = 0

            if self.elitism:
                start_index = 1

            children = [
                parent.reproduce(*children_args)
                for _ in range(self.num_children)
                for parent in population[start_index:]
            ]

            if self.elitism:
                children.append(best.copy())

            sorted_combined = sorted(
                children,
                reverse=self.maximize,
                key=lambda individual: individual.evaluate(self.fitness_function),
            )

            if self.maximize:
                if sorted_combined[0].fitness > best.fitness:
                    best = sorted_combined[0].copy()
            else:  # Minimize
                if sorted_combined[0].fitness < best.fitness:
                    best = sorted_combined[0].copy()

            self.evaluations += len(population)
            population = sorted_combined[: self.population_size]
            best = population[0]

            if self.reporter is not None:
                mean = np.mean([x.fitness for x in population])
                std = np.std([x.fitness for x in population])
                self.reporter(
                    ProgressReport(
                        generation,
                        self.evaluations,
                        best.genotype,
                        best.fitness,
                        mean,
                        std,
                    )
                )

            if self._check_early_stop(start_time, best):
                break

        return best.genotype

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
