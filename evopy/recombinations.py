from enum import Enum


class RecombinationStrategy(str, Enum):
    NONE = "none"
    WEIGHTED = "weighted"
    INTERMEDIATE = "intermediate"
    CORRELATED_MUTATIONS = "correlated_mutations"

    @staticmethod
    def from_string(s: str):
        try:
            return RecombinationStrategy(s)
        except ValueError:
            ValueError(f"Invalid recombination strategy: {s}")

    def __str__(self):
        return self.value
