"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation.

Based on "The CMA Evolution Strategy: A Comparing Review" pp 75–102
https://link.springer.com/chapter/10.1007/3-540-32494-1_4
"""

import numpy as np
from evopy.individual import Individual
from evopy.strategy import Strategy
from evopy.utils import random_with_seed


class CMAState:
    """Holds the state for CMA-ES algorithm."""

    def __init__(self, n_dimensions, mu, random_seed=None):
        self.mean = None  # Population mean
        self.covariance_matrix = np.identity(n_dimensions)  # Covariance matrix
        self.sigma = 1.0  # Overall step size
        self.evolution_path = 0  # Evolution path for step size adaptation
        self.mu = mu
        self.random_seed = random_seed
        self.random = random_with_seed(random_seed)

    def reproduce_cma(
        self,
        population,
        lambd,
        fitness_function,
        constraint_handling_func,
        bounds,
    ):
        """Perform one generation of CMA-ES.

        Args:
            population: Current population
            lambd: Number of offspring to generate
            fitness_function: Function to evaluate fitness
            constraint_handling_func: Function to handle constraints
            bounds: Bounds for the parameters

        Returns:
            List of new individuals for next generation
        """
        assert self.mu <= len(
            population
        ), "Cannot select more individuals than present in the population"
        # Initialize state if this is the first generation
        if self.mean is None:
            xs = np.array([x.genotype for x in population])
            self.mean = np.mean(xs[: self.mu], axis=0)

        # Generate new samples using current distribution
        xs = [
            self.mean
            + self.sigma
            * self.random.multivariate_normal(
                np.zeros(self.mean.shape[0]), self.covariance_matrix
            )
            for _ in range(lambd)
        ]

        # Create and evaluate new individuals
        next_population = [
            Individual(
                x,
                Strategy.CMA,
                strategy_parameters=None,
                constraint_handling_func=constraint_handling_func,
                bounds=bounds,
                random_seed=self.random_seed,
            )
            for x in xs
        ]
        for x in next_population:
            x.genotype = constraint_handling_func(x, x.genotype)

        evaluated = [(x, x.evaluate(fitness_function)) for x in next_population]
        evaluated.sort(key=lambda tup: tup[1], reverse=True)

        top_mu = evaluated[: self.mu]
        xs_top = np.array([ind.genotype for ind, _ in top_mu])

        # Calculate weights for ranked individuals
        ranks = np.arange(1, self.mu + 1)
        weights = np.log(self.mu + 0.5) - np.log(ranks)
        weights = weights / np.sum(weights)

        # === Update CMA-ES state ===
        old_mean = self.mean.copy()
        n = self.mean.shape[0]

        # Update hyperparameters
        mu_eff = 1.0 / np.sum(weights**2)  # Variance effective selection mass
        c_sigma = np.sqrt(mu_eff + 2) / (
            n + np.sqrt(mu_eff + 2)
        )  # Learning rate for step size
        d_sigma = (
            1.0 + 2.0 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        )  # Damping for step size
        c_cov = 4.0 / (n + 4.0)  # Learning rate for covariance matrix

        # Update mean
        self.mean = np.sum(weights[:, np.newaxis] * xs_top, axis=0)

        # Ensure positive definiteness
        eigenvalues, _ = np.linalg.eigh(self.covariance_matrix)
        assert np.all(eigenvalues > 0), "Covariance matrix must be positive definite"

        # Update evolution path and covariance matrix
        C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(self.covariance_matrix))
        step = (self.mean - old_mean) / self.sigma
        self.evolution_path = (1 - c_sigma) * self.evolution_path + np.sqrt(
            c_sigma * (2 - c_sigma) * mu_eff
        ) * (C_inv_sqrt @ step)

        # Rank-μ update combined with evolution path
        self.covariance_matrix = (
            (1 - c_cov) * self.covariance_matrix
            + (c_cov / mu_eff)  # Decay of old matrix
            * np.outer(self.evolution_path, self.evolution_path)
            + c_cov  # Evolution path update
            * (1 - 1 / mu_eff)
            * np.sum(
                [  # Rank-μ update
                    weights[i]
                    * np.outer(
                        (xs_top[i] - old_mean) / self.sigma,
                        (xs_top[i] - old_mean) / self.sigma,
                    )
                    for i in range(self.mu)
                ],
                axis=0,
            )
        )

        # Update step size using evolution path length
        self.sigma = self.sigma * np.exp(
            (c_sigma / d_sigma)
            * ((np.linalg.norm(self.evolution_path) / expected_norm(n)) - 1)
        )
        # Prevent numerical issues with step size
        self.sigma = np.clip(self.sigma, 1e-8, 1.0)

        return next_population


def expected_norm(n):
    """Calculate the expected norm of a n-dimensional normal distribution.
    Used for step size adaptation in CMA-ES."""
    return np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * (n**2)))
