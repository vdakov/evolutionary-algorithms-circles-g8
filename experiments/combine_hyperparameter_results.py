import os
import json
import sys
import numpy as np
from typing import Dict, List, Any


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    results = []
    for filename in os.listdir(results_dir):
        if filename.startswith("results") and filename.endswith(".json"):
            with open(os.path.join(results_dir, filename), "r") as f:
                results.append(json.load(f))
    return results


def get_config_key(config: Dict[str, Any]) -> str:
    key_params = [
        # "generations",
        # "n_circles",
        # "num_children",
        "population_size",
        # "variance_strategy",
        # "constraint_handling",
        # "elitism",
        "initialization_strategy",
        # "recombination_strategy",
        "jitter",
    ]
    # Create a tuple of parameter values in a fixed order
    key_values = []
    for param in key_params:
        value = config[param]
        # Convert to string if it's not a basic type
        if not isinstance(value, (int, float, str, bool)):
            value = str(value)
        key_values.append(f"{param}:{value}")

    return ",".join(key_values)


def combine_and_sort_results(results_dir: str) -> List[Dict[str, Any]]:
    # Load all results
    all_results = load_results(results_dir)

    # Group results by configuration (ignoring num_children)
    config_groups = {}
    for result in all_results:
        key = get_config_key(result)
        if key not in config_groups:
            config_groups[key] = []
        config_groups[key].append(result)

    # Combine results for each configuration group
    combined_results = []
    for key, group in config_groups.items():
        # Collect all runs from all configurations in the group
        all_runs = []
        mean_fitnesses = []
        best_fitnesses = []
        for config in group:
            all_runs.append(len(config["runs"]))
            mean_fitnesses.append(config["mean_fitness"])
            best_fitnesses.append(config["best_fitness"])

        combined_result = {
            "configuration": key,
            "best_fitness": float(max(best_fitnesses)),
            "mean_fitness": float(np.mean(mean_fitnesses)),
            "n_total_runs": int(np.sum(all_runs)),
        }
        combined_results.append(combined_result)

    # Sort by mean fitness (descending)
    sorted_results = sorted(
        combined_results, key=lambda x: x["mean_fitness"], reverse=True
    )
    return sorted_results


if __name__ == "__main__":
    results_dir = sys.argv[1]
    print(f"Analyzing results from: {results_dir}")
    # Combine and sort results
    sorted_results = combine_and_sort_results(results_dir)
    # Save sorted results
    output_file = os.path.join(results_dir, "combined_sorted_results.json")
    with open(output_file, "w") as f:
        json.dump(sorted_results, f, indent=4)
