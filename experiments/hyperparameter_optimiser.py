import math
import multiprocess
import os
import time
import json
from datetime import datetime
from evopy import Strategy, ConstraintHandling, InitializationStrategy, ResultsManager
from evopy.recombinations import RecombinationStrategy
from experiments import run_comparison
from problem_runner import CirclesInASquare
import numpy as np

# Performs a gridsearch over most hyperparameters
if __name__ == "__main__":

    result_directory = os.path.join(
        "outputs", f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_hyperparameter_optimiser'
    )
    os.makedirs(result_directory, exist_ok=True)

    # format: (name, possible value)
    options = [
        ("generations", [2000]),
        ("n_circles", [15]),
        ("population_size", [10, 30, 50, 100]),
        ("num_children", [1]), # irrelevant
        ("variance_strategy", [Strategy.CMA]),
        ("constraint_handling", [ConstraintHandling.BOUNDARY_REPAIR]),
        ("with_elitism", [False]), # irrelevant
        ("initialization_strategy", [s for s in InitializationStrategy]),
        ("recombination_strategy", [RecombinationStrategy.NONE]),
        ("jitter", [0, 0.1]),
        ("print_solutions", [False]),
        ("plot_solutions", [False]),
    ]

    max_evals =  1e7
    n_runs = 10 # runs per settings to be averaged
    seeds = np.random.randint(0, 1000000, size=n_runs)
    seeds = [609552, 529194, 429445,  84287, 363599, 113265, 423712,  42939, 932637, 679017]
    parallelize = False
    print(f"Seeds used: {seeds}")

    total_experiments = math.prod([len(x[1]) for x in options])
    current_experiment = 0
    indexes = np.zeros((len(options)), dtype=int)
    while indexes is not None:
        current_experiment += 1
        # select the current parameters
        current_parameters = {}
        for i in range(len(options)):
            current_parameters[options[i][0]] = options[i][1][indexes[i]]
        print(f"Starting experiment {current_experiment}/{total_experiments}")
        print(f"Settings: {current_parameters}")
        start_time = time.time()

        results_manager = ResultsManager(
            f"hyperparameter_optimisation_{current_experiment}", save_files=False
        )

        def do_instance(instance_seed, current_parameters, results_manager, max_evals):
            # create instance
            runner = CirclesInASquare(
                n_circles=current_parameters['n_circles'],
                init_strategy=InitializationStrategy.from_string(current_parameters['initialization_strategy']),
                print_sols=current_parameters["print_solutions"],
                plot_sols=current_parameters["plot_solutions"],
                output_statistics=True,
                init_jitter=current_parameters['jitter'],
                results_manager=results_manager,
                random_seed=instance_seed,
                print_header=False
            )
            best_solution = runner.run_evolution_strategies(
                population_size=current_parameters['population_size'],
                num_children=current_parameters['num_children'],
                generations=current_parameters['generations'],
                strategy=Strategy.from_string(current_parameters['variance_strategy']),
                constraint_handling=ConstraintHandling.from_string(current_parameters['constraint_handling']),
                max_evaluations=max_evals,
                recombination_strategy=RecombinationStrategy.from_string(current_parameters['recombination_strategy']),
                elitism=current_parameters['with_elitism'],
                max_run_time=None,
            )
            return {
                    "seed": int(instance_seed),
                    "best_solution": (
                        best_solution.tolist()
                        if isinstance(best_solution, np.ndarray)
                        else best_solution
                    ),
                    "best_fitness": (
                        runner.best_total_score[-1]
                        if runner.best_total_score
                        else None
                    ),
                    "target_value": runner.get_target(),
                    "generations_run": len(runner.best_total_score),
                    "progression": runner.best_total_score,
                }

        results = []
        if parallelize:
            with multiprocess.Pool(len(seeds)) as pool:
                 results = pool.starmap(do_instance, [(s, current_parameters, results_manager, max_evals) for s in seeds])
        else:
            for seed in seeds:
                results.append(do_instance(seed, current_parameters, results_manager, max_evals))

        end_time = time.time()
        print(f"Finished experiment {current_experiment}/{total_experiments} in {end_time - start_time} seconds")

        # reuse current_parameters value to use for writing to file without copying the parameters
        fitnesses = [r['best_fitness'] for r in results]
        current_parameters["total_time"] = end_time - start_time
        current_parameters["mean_fitness"] = np.mean(fitnesses)
        current_parameters["std_fitness"] = np.std(fitnesses)
        current_parameters["best_fitness"] = np.max(fitnesses)
        current_parameters["worst_fitness"] = np.min(fitnesses)
        current_parameters['runs'] = results
        with open(os.path.join(result_directory, f"results {str(current_experiment).zfill(len(str(total_experiments)))}.json"), "w") as f:
            json.dump(current_parameters, f, indent=4)

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

