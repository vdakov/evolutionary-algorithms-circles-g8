# Example adjustment if needed:
class Strategy:
    # Possible strategies
    SINGLE = 0
    MULTIPLE = 1
    FULL_VARIANCE = 2

    @staticmethod
    def from_string(strategy_str):
        # Map strings to enum values
        if strategy_str == "single":
            return Strategy.SINGLE
        elif strategy_str == "multiple":
            return Strategy.MULTIPLE
        elif strategy_str == "full":
            return Strategy.FULL_VARIANCE
        else:
            raise ValueError(f"Invalid strategy: {strategy_str}")
