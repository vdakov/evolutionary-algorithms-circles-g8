import argparse
from enum import Enum

import numpy as np

from evopy.individual import Individual


class ConstraintHandling(Enum):
    """Enum used to distinguish different types of resolving individuals that violate the bounds constraints.

    These strategies are used to determine the mechanism which each individual refers to when a constraint is violated.

    - BOUNDARY_REPAIR: If x_i < 0 set it to 0; if x_i > 1 set it to 1
    - CONSTRAINT_DOMINATION: Allow violations but penalize them
    - RANDOM_REPAIR: If x_i < 0 or x_i > 1, resample x_i
    """

    BOUNDARY_REPAIR = 1
    CONSTRAINT_DOMINATION = 2
    RANDOM_REPAIR = 3

    @staticmethod
    def from_string(s: str):
        match s:
            case "BR":
                return ConstraintHandling.BOUNDARY_REPAIR
            case "CD":
                return ConstraintHandling.CONSTRAINT_DOMINATION
            case "RR":
                return ConstraintHandling.RANDOM_REPAIR
            case _:
                argparse.ArgumentTypeError(
                    f"Invalid Constraint Handling Technique: {s}"
                )


def run_boundary_repair(individual: Individual, new_genotype):
    # Clip values to bounds
    new_genotype = np.maximum(new_genotype, individual.bounds[0])
    new_genotype = np.minimum(new_genotype, individual.bounds[1])
    return new_genotype


def run_constraint_domination(individual: Individual, new_genotype):
    raise NotImplementedError()


def run_random_repair(individual: Individual, new_genotype):
    # Randomly sample out-of-bounds indices
    oob_indices = (new_genotype < individual.bounds[0]) | (
        new_genotype > individual.bounds[1]
    )
    new_genotype[oob_indices] = individual.random.uniform(
        individual.bounds[0], individual.bounds[1], size=np.count_nonzero(oob_indices)
    )
    return new_genotype
