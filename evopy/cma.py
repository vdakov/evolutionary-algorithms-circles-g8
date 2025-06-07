"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation.

Based on "The CMA Evolution Strategy: A Comparing Review" pp 75-102
https://link.springer.com/chapter/10.1007/3-540-32494-1_4
"""

import numpy as np
from evopy.individual import Individual
from evopy.strategy import Strategy
from evopy.utils import random_with_seed


class CMAState:
    """Holds the state for CMA-ES algorithm."""

    def __init__(self, n_dimensions, mu, random_seed=None):
        self.n = n_dimensions
        self.mu = mu

        # Initialize dynamic state
        self.mean = None  # Population mean
        self.covariance_matrix = np.identity(n_dimensions)  # Covariance matrix
        self.sigma = 1.0  # Overall step size
        self.evolution_path = np.zeros(
            n_dimensions
        )  # Evolution path for step size adaptation

        # Pre-compute constant parameters
        # Calculate weights for ranked individuals (equation 4)
        ranks = np.arange(1, mu + 1)
        self.weights = np.log(mu + 0.5) - np.log(ranks)
        self.weights = self.weights / np.sum(self.weights)

        # Variance effective selection mass (equation 5)
        self.mu_eff = 1.0 / np.sum(self.weights**2)

        # Default parameters for CMA-ES (Table 2)
        # Learning rate for step size
        self.c_sigma = np.sqrt(self.mu_eff + 2) / (
            n_dimensions + np.sqrt(self.mu_eff + 2)
        )
        # Damping for step size
        self.d_sigma = (
            1.0
            + 2.0 * max(0, np.sqrt((self.mu_eff - 1) / (n_dimensions + 1)) - 1)
            + self.c_sigma
        )
        # Learning rate for covariance matrix
        self.c_cov = 4.0 / (n_dimensions + 4.0)

        # Pre-compute constants for evolution path update (equation 17)
        self.cs_factor = np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff)
        self.one_minus_c_sigma = 1 - self.c_sigma

        # Pre-compute constants for covariance matrix update (equation 22)
        self.one_minus_c_cov = 1 - self.c_cov
        self.c_cov_mu_eff = self.c_cov / self.mu_eff
        self.c_cov_one_minus_mu_eff = self.c_cov * (1 - 1 / self.mu_eff)

        # Pre-compute expected norm for step size adaptation
        self.chiN = expected_norm(n_dimensions)

        # Random state
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
        # Generate new samples using current distribution (equation 2)
        xs = self.random.multivariate_normal(
            self.mean, (self.sigma**2) * self.covariance_matrix, size=lambd
        )
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
        # Apply constraints
        for x in next_population:
            x.genotype = constraint_handling_func(x, x.genotype)
        # Evaluate and sort by fitness
        evaluated = [(x, x.evaluate(fitness_function)) for x in next_population]
        evaluated.sort(key=lambda tup: tup[1], reverse=True)
        # Select the best mu individuals and extract their genotypes
        xs_top = np.array([ind.genotype for ind, _ in evaluated[: self.mu]])

        # === Update CMA-ES state ===
        old_mean = self.mean.copy()
        # Update mean (equation 4)
        # Vectorized weighted sum
        self.mean = np.sum(self.weights[:, np.newaxis] * xs_top, axis=0)

        # Ensure positive definiteness
        eigenvalues = np.linalg.eigvalsh(self.covariance_matrix)
        assert np.all(eigenvalues > 0), "Covariance matrix must be positive definite"

        # Update evolution path (equation 17)
        C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(self.covariance_matrix))
        step = (self.mean - old_mean) / self.sigma
        self.evolution_path = (
            self.one_minus_c_sigma * self.evolution_path
            + self.cs_factor * (C_inv_sqrt @ step)
        )

        # Rank-μ update combined with evolution path (equation 22)
        diff_vectors = (xs_top - old_mean[np.newaxis, :]) / self.sigma
        weighted_covs = np.sum(
            [w * np.outer(d, d) for w, d in zip(self.weights, diff_vectors)], axis=0
        )
        self.covariance_matrix = (
            self.one_minus_c_cov * self.covariance_matrix
            + self.c_cov_mu_eff * np.outer(self.evolution_path, self.evolution_path)
            + self.c_cov_one_minus_mu_eff * weighted_covs
        )

        # Update step size using evolution path length (equation 28)
        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma)
            * (np.linalg.norm(self.evolution_path) / self.chiN - 1)
        )
        # Prevent numerical issues with step size
        self.sigma = np.clip(self.sigma, 1e-8, 1.0)

        return next_population


def expected_norm(n):
    """Calculate the expected norm of a n-dimensional normal distribution
    for step size adaptation. Uses Taylor expansion, 2nd order"""
    return np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * (n**2)))
