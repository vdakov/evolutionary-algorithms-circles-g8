class ProgressReport:
    """Class representing a report on an intermediate state of the learning process."""

    def __init__(
        self,
        generation,
        evaluations,
        best_genotype,
        best_fitness,
        avg_fitness,
        std_fitness,
        best_genotype_so_far,
    ):
        """Initializes the report instance.

        :param generation: number identifying the reported generation
        :param best_genotype: the genotype of the best individual of that generation
        :param best_fitness: the fitness of the best individual of that generation
        """
        self.generation = generation
        self.evaluations = evaluations
        self.best_genotype = best_genotype
        self.best_fitness = best_fitness
        self.avg_fitness = avg_fitness
        self.std_fitness = std_fitness
        self.best_genotype_so_far = best_genotype_so_far
