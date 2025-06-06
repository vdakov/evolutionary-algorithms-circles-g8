from enum import Enum


class Strategy(Enum):
    SINGLE = "single"
    MULTIPLE = "multiple"
    FULL_VARIANCE = "full"

    @staticmethod
    def from_string(s: str):
        try:
            return Strategy(s)
        except ValueError:
            ValueError(f"Invalid strategy: {s}")

    def __str__(self):
        return self.value
