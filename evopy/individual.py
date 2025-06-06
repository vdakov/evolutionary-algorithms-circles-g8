import numpy as np

from evopy.strategy import Strategy
from evopy.utils import random_with_seed


class Individual:
    """The individual of the evolutionary strategy algorithm.

    This class handles the reproduction of the individual, using both the genotype and the specified
    strategy.

    For the full variance reproduction strategy, we adopt the implementation as described in:
    [1] Schwefel, Hans-Paul. (1995). Evolution Strategies I: Variants and their computational
        implementation. G. Winter, J. Perieaux, M. Gala, P. Cuesta (Eds.), Proceedings of Genetic
        Algorithms in Engineering and Computer Science, John Wiley & Sons.
    """

    _BETA = 0.0873
    _EPSILON = 0.01

    def __init__(
        self,
        genotype,
        strategy,
        strategy_parameters,
        constraint_handling_func,
        age=0,
        bounds=None,
        random_seed=None,
    ):
        """Initialize the Individual.

        :param genotype: the genotype of the individual
        :param strategy: the strategy chosen to reproduce. See the Strategy enum for more
                         information
        :param strategy_parameters: the parameters required for the given strategy, as a list
        """
        self.genotype = genotype
        self.length = len(genotype)
        self.random_seed = random_seed
        self.random = random_with_seed(self.random_seed)
        self.fitness = None
        self.age = age
        self.constraint = None
        self.bounds = bounds
        self.strategy = strategy
        self.strategy_parameters = np.asarray(strategy_parameters)
        self.constraint_handling_func = constraint_handling_func

        if strategy == Strategy.SINGLE and len(strategy_parameters) == 1:
            self.reproduce = self._reproduce_single_variance
        elif strategy == Strategy.MULTIPLE and len(strategy_parameters) == self.length:
            self.reproduce = self._reproduce_multiple_variance
        elif (
            strategy == Strategy.FULL_VARIANCE
            and len(strategy_parameters) == self.length * (self.length + 1) / 2
        ):
            self.reproduce = self._reproduce_full_variance
        else:
            raise ValueError("The length of the strategy parameters was not correct.")

    def evaluate(self, fitness_function):
        """Evaluate the genotype of the individual using the provided fitness function.

        :param fitness_function: the fitness function to evaluate the individual with
        :return: the value of the fitness function using the individuals genotype
        """
        self.fitness = fitness_function(self.genotype)

        return self.fitness

    def _reproduce_single_variance(
        self,
        weights=None,
        population=None,
        reproduction_strategy=None,
        correlated_mutations=False,
    ):
        """Create a single offspring individual from the set genotype and strategy parameters.

        This function uses the single variance strategy.

        :return: an individual which is the offspring of the current instance
        """
        used_genotype, used_strategy_parameters = (
            self.genotype,
            self.strategy_parameters,
        )

        if reproduction_strategy:
            recombination_params = (weights, population)
            used_genotype, used_strategy_parameters = self.recombination(
                self.genotype,
                self.strategy_parameters,
                recombination_params,
                reproduction_strategy=reproduction_strategy,
            )

        new_genotype = used_genotype + used_strategy_parameters[0] * self.random.randn(
            self.length
        )
        new_genotype = self.constraint_handling_func(self, new_genotype)
        scale_factor = self.random.randn() * np.sqrt(1 / (2 * self.length))
        new_parameters = [
            max(used_strategy_parameters[0] * np.exp(scale_factor), self._EPSILON)
        ]
        return Individual(
            new_genotype,
            self.strategy,
            new_parameters,
            self.constraint_handling_func,
            bounds=self.bounds,
            random_seed=self.random,
        )

    def _reproduce_multiple_variance(
        self,
        weights=None,
        population=None,
        reproduction_strategy=None,
        correlated_mutations=False,
    ):
        """Create a single offspring individual from the set genotype and strategy.

        This function uses the multiple variance strategy.

        :return: an individual which is the offspring of the current instance
        """
        used_genotype, used_strategy_parameters = (
            self.genotype,
            self.strategy_parameters,
        )

        if reproduction_strategy:
            recombination_params = (weights, population)
            used_genotype, used_strategy_parameters = self.recombination(
                self.genotype,
                self.strategy_parameters,
                recombination_params,
                reproduction_strategy=reproduction_strategy,
            )

        new_genotype = used_genotype + [
            used_strategy_parameters[i] * self.random.randn()
            for i in range(self.length)
        ]
        new_genotype = self.constraint_handling_func(self, new_genotype)
        global_scale_factor = self.random.randn() * np.sqrt(1 / (2 * self.length))
        scale_factors = [
            self.random.randn() * np.sqrt(1 / 2 * np.sqrt(self.length))
            for _ in range(self.length)
        ]

        new_parameters = [
            max(
                np.exp(global_scale_factor + scale_factors[i])
                * self.strategy_parameters[i],
                self._EPSILON,
            )
            for i in range(self.length)
        ]

        return Individual(
            new_genotype,
            self.strategy,
            new_parameters,
            self.constraint_handling_func,
            bounds=self.bounds,
        )

    # pylint: disable=invalid-name
    def _reproduce_full_variance(
        self, weights=None, population=None, reproduction_strategy=None
    ):
        """Create a single offspring individual from the set genotype and strategy.

        This function uses the full variance strategy, as described in [1]. To emphasize this, the
        variable names of [1] are used in this function.

        :return: an individual which is the offspring of the current instance
        """

        used_genotype, used_strategy_parameters = (
            self.genotype,
            self.strategy_parameters,
        )

        if reproduction_strategy:
            recombination_params = (weights, population)
            used_genotype, used_strategy_parameters = self.recombination(
                self.genotype,
                self.strategy_parameters,
                recombination_params,
                reproduction_strategy=reproduction_strategy,
            )

        global_scale_factor = self.random.randn() * np.sqrt(1 / (2 * self.length))
        scale_factors = [
            self.random.randn() * np.sqrt(1 / 2 * np.sqrt(self.length))
            for _ in range(self.length)
        ]

        new_variances = [
            max(
                np.exp(global_scale_factor + scale_factors[i])
                * used_strategy_parameters[i],
                self._EPSILON,
            )
            for i in range(self.length)
        ]
        new_rotations = [
            used_strategy_parameters[i] + self.random.randn() * self._BETA
            for i in range(self.length, len(used_strategy_parameters))
        ]
        new_rotations = [
            (
                rotation
                if abs(rotation) < np.pi
                else rotation - np.sign(rotation) * 2 * np.pi
            )
            for rotation in new_rotations
        ]
        T = np.identity(self.length)
        for p in range(self.length - 1):
            for q in range(p + 1, self.length):
                j = int((2 * self.length - p) * (p + 1) / 2 - 2 * self.length + q)
                T_pq = np.identity(self.length)
                T_pq[p][p] = T_pq[q][q] = np.cos(new_rotations[j])
                T_pq[p][q] = -np.sin(new_rotations[j])
                T_pq[q][p] = -T_pq[p][q]
                T = np.matmul(T, T_pq)
        new_genotype = used_genotype + T @ self.random.randn(self.length)
        new_genotype = self.constraint_handling_func(self, new_genotype)

        strategy_parameters = np.array(new_variances + new_rotations)

        return Individual(
            new_genotype,
            self.strategy,
            strategy_parameters,
            self.constraint_handling_func,
            bounds=self.bounds,
        )

    def recombination(
        self, x, strategy_parameters, recombination_params, reproduction_strategy
    ):
        if reproduction_strategy == "weighted":
            weights, population = recombination_params
            w_x = 0.7
            w_r = 0.3
            aggregate_x = np.zeros(x.shape)
            aggregate_r = np.zeros(strategy_parameters.shape)
            for i, individual in enumerate(population):
                aggregate_x += individual.genotype * weights[i]
                aggregate_r += individual.strategy_parameters * weights[i]

            x_new = (w_x) * x + aggregate_x * w_r
            strategy_parameters = (w_x) * strategy_parameters + aggregate_r * w_r
            return x_new, strategy_parameters
        elif reproduction_strategy == "intermediate":
            weights, population = recombination_params

            if not population:
                return  # Nothing to recombine if no parents provided

            num_parents = len(population)

            # Initialize sums for genotype and strategy parameters
            sum_genotype = np.zeros(x.shape)
            # sum_strategy_parameters = np.zeros(self.strategy_parameters.shape)

            # Sum up all parent contributions
            for individual in population:
                sum_genotype += individual.genotype
                # sum_strategy_parameters += individual.strategy_parameters

            # Compute the average (centroid) and update
            average_genotype = sum_genotype / num_parents
            # average_strategy_parameters = sum_strategy_parameters / num_parents
            return average_genotype, self.strategy_parameters
        elif reproduction_strategy == "correlated_mutations":
            _, population = recombination_params

            # pick random individual to combine with
            other_individual = population[self.random.randint(0, len(population))]
            average_genotype = (self.genotype + other_individual.genotype) / 2.0
            average_strategy_parameters = (
                self.strategy_parameters + other_individual.strategy_parameters
            ) / 2.0
            return average_genotype, average_strategy_parameters
        else:
            raise ValueError(f"Unknown recombination type: {reproduction_strategy}")

    def copy(self):
        """
        Creates a deep copy of the Individual object.
        """
        # Ensure deep copies of mutable attributes like genotype and strategy_parameters
        new_individual = Individual(
            genotype=np.copy(self.genotype),
            strategy=self.strategy,  # Enum or immutable, so direct assignment is fine
            strategy_parameters=np.copy(
                self.strategy_parameters
            ),  # Copy the strategy parameters
            constraint_handling_func=self.constraint_handling_func,  # Function, so direct assignment is fine
            bounds=self.bounds,  # Tuple/immutable, so direct assignment is fine
            age=self.age,
            random_seed=self.random_seed,
        )
        # Copy the cached fitness and age
        new_individual.fitness = self.fitness
        new_individual.age = self.age
        return new_individual
