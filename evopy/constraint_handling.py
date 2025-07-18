import numpy as np
from enum import Enum

from evopy.individual import Individual


class ConstraintHandling(str, Enum):
    """Enum used to distinguish different types of resolving individuals that violate the bounds constraints.

    These strategies are used to determine the mechanism which each individual refers to when a constraint is violated.

    - BOUNDARY_REPAIR: If x_i < 0 set it to 0; if x_i > 1 set it to 1
    - CONSTRAINT_DOMINATION: Allow violations but penalize them
    - RANDOM_REPAIR: If x_i < 0 or x_i > 1, resample x_i
    """

    BOUNDARY_REPAIR = "BR"
    CONSTRAINT_DOMINATION = "CD"
    RANDOM_REPAIR = "RR"

    @staticmethod
    def from_string(s: str):
        try:
            return ConstraintHandling(s)
        except ValueError:
            raise ValueError(f"Invalid Constraint Handling Technique: {s}")

    def __str__(self):
        return self.value


def run_boundary_repair(individual: Individual, new_genotype):
    # Clip values to bounds
    new_genotype = np.maximum(new_genotype, individual.bounds[0])
    new_genotype = np.minimum(new_genotype, individual.bounds[1])
    return new_genotype


def run_constraint_domination(individual: Individual, new_genotype):
    lower_violations = np.maximum(individual.bounds[0] - new_genotype, 0) ** 2
    upper_violations = np.maximum(new_genotype - individual.bounds[1], 0) ** 2

    total_violation = lower_violations + upper_violations

    violation_error = np.sum(total_violation)

    individual.violation_error = violation_error
    return np.clip(new_genotype, individual.bounds[0], individual.bounds[1])


def run_random_repair(individual: Individual, new_genotype):
    # Randomly sample out-of-bounds indices
    oob_indices = (new_genotype < individual.bounds[0]) | (
        new_genotype > individual.bounds[1]
    )
    new_genotype[oob_indices] = individual.random.uniform(
        individual.bounds[0], individual.bounds[1], size=np.count_nonzero(oob_indices)
    )
    return new_genotype


constraint_func_dispatch = {
    ConstraintHandling.BOUNDARY_REPAIR: run_boundary_repair,
    ConstraintHandling.CONSTRAINT_DOMINATION: run_constraint_domination,
    ConstraintHandling.RANDOM_REPAIR: run_random_repair,
}
