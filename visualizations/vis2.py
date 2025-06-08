import json
import glob
import matplotlib.pyplot as plt

json_files = glob.glob("C:\\Users\\Todor\\Desktop\\results\\results *.json")

final_fitness_by_strategy = {}

for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        if data["population_size"] == 100:
            init = data["initialization_strategy"]
            final_fitness_by_strategy.setdefault(init, [])
            for run in data["runs"]:
                final_fitness_by_strategy[init].append(run["best_fitness"])

plt.figure(figsize=(10, 6))
strategies = list(final_fitness_by_strategy.keys())
fitness_data = [final_fitness_by_strategy[init] for init in strategies]

plt.boxplot(fitness_data, labels=strategies, showmeans=True)
plt.ylabel("Final Best Fitness")
plt.title("Final Fitness Distribution by Initialization Strategy (Population = 100)")
plt.grid(axis="y")
plt.tight_layout()
plt.show()
