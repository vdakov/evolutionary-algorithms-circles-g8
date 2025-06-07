import numpy as np
from evopy.recombinations import RecombinationStrategy
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
        elif strategy == Strategy.CMA:
            self.reproduce = None  # Handled separately
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
        weights, population = recombination_params
        if reproduction_strategy == RecombinationStrategy.WEIGHTED.value:
            w_x = 0.7  # Weight for current individual
            w_r = 0.3  # Weight for population influence
            # Compute weighted sum using numpy for efficiency
            pop_genotypes = np.array([ind.genotype for ind in population])
            pop_strategies = np.array([ind.strategy_parameters for ind in population])
            aggregate_x = np.sum(pop_genotypes * weights[:, np.newaxis], axis=0)
            aggregate_r = np.sum(pop_strategies * weights[:, np.newaxis], axis=0)
            x_new = w_x * x + w_r * aggregate_x
            strategy_parameters_new = w_x * strategy_parameters + w_r * aggregate_r
            return x_new, strategy_parameters_new
        elif reproduction_strategy == RecombinationStrategy.INTERMEDIATE.value:
            if not population:
                return x, strategy_parameters
            # Efficient intermediate recombination using numpy
            pop_genotypes = np.array([ind.genotype for ind in population])
            average_genotype = np.mean(pop_genotypes, axis=0)
            # Keep strategy parameters from parent to maintain adaptation
            return average_genotype, strategy_parameters
        elif reproduction_strategy == RecombinationStrategy.CORRELATED_MUTATIONS.value:
            # Improved correlated mutation with random parent selection
            other_individual = population[self.random.randint(0, len(population))]
            # Adaptive mixing based on fitness difference
            if self.fitness is not None and other_individual.fitness is not None:
                # More weight to better parent
                alpha = 0.5 + 0.3 * (
                    1 if self.fitness > other_individual.fitness else -1
                )
            else:
                alpha = 0.5
            x_new = alpha * x + (1 - alpha) * other_individual.genotype
            strategy_parameters_new = (
                alpha * strategy_parameters
                + (1 - alpha) * other_individual.strategy_parameters
            )
            return x_new, strategy_parameters_new
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
