from evopy import Strategy, ConstraintHandling, InitializationStrategy, ResultsManager
from evopy.recombinations import RecombinationStrategy
from experiments import run_comparison
from problem_runner import CirclesInASquare
import numpy as np

# Performs a gridsearch over most hyperparameters
if __name__ == "__main__":
    # format: (name, possible value)
    options = [
        ("population_size", [10, 30, 50, 100, 250]),
        ("num_children", [1, 2, 4, 8]),
        ("variance_strategy", [Strategy.CMA]),
        ("constraint_handling", [ConstraintHandling.BOUNDARY_REPAIR]),
        ("with_elitism", [True, False]),
        ("initialization_strategy", [s for s in InitializationStrategy]),
        ("recombination_strategy", [RecombinationStrategy.NONE]),
        ("jitter", [0, 0.1, 0.25]),
    ]

    generations = 500
    n_circles = 10
    print_solutions = False
    plot_solutions = False
    max_evals =  1e7
    n_runs = 10 # runs per settings to be averaged

    indexes = np.zeros((len(options)), dtype=int)
    while indexes is not None:
        # select the current parameters
        current_parameters = {}
        for i in range(len(options)):
            current_parameters[options[i][0]] = options[i][1][indexes[i]]

        # create instance
        runner = CirclesInASquare(
            n_circles=n_circles,
            init_strategy=InitializationStrategy.from_string(current_parameters['initialization_strategy']),
            print_sols=print_solutions,
            plot_sols=plot_solutions,
            init_jitter=current_parameters['jitter'],
            results_manager=ResultsManager(),
            random_seed=None, # TODO
        )
        best = runner.run_evolution_strategies(
            population_size=current_parameters['population_size'],
            num_children=current_parameters['num_children'],
            generations=generations,
            strategy=Strategy.from_string(current_parameters['variance_strategy']),
            constraint_handling=ConstraintHandling.from_string(current_parameters['constraint_handling']),
            max_evaluations=max_evals,
            recombination_strategy=RecombinationStrategy.from_string(current_parameters['recombination_strategy']),
            elitism=current_parameters['with_elitism'],
            max_run_time=None,
        )


        # increment the indexes
        for i in range(len(indexes)):
            # increment index, if we went through all options, loop and increment the next and so on
            indexes[i] += 1
            if indexes[i] != len(options[i][1]):
                break
            indexes[i] = 0

            # if we went to the last option, we exit the outer loop
            if i == len(indexes) - 1:
                indexes = None


    # random_seed = ???

    # results, analysis = run_comparison(
    #     "Recombination Strategy",
    #     options=[r for r in RecombinationStrategy],
    #     param_to_overwrite="recombination_strategy",
    #     param_in_runner=True,
    #     n_circles=10,
    #     n_runs=5,
    #     population_size=30,
    #     num_children=1,
    #     generations=100,
    #     with_elitism=True,
    # )

