import json
import glob
import matplotlib.pyplot as plt

json_files = glob.glob("C:\\Users\\Todor\\Desktop\\results\\results *.json")

final_fitness_by_popsize = {}

for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        pop_size = data["population_size"]
        final_fitness_by_popsize.setdefault(pop_size, [])
        for run in data["runs"]:
            final_fitness_by_popsize[pop_size].append(run["best_fitness"])

final_fitness_by_popsize = {
    k: v for k, v in final_fitness_by_popsize.items() if len(v) > 0
}

sorted_pop_sizes = sorted(final_fitness_by_popsize.keys())
fitness_data = [final_fitness_by_popsize[pop] for pop in sorted_pop_sizes]

plt.figure(figsize=(10, 6))
plt.boxplot(fitness_data, tick_labels=sorted_pop_sizes)
plt.xlabel("Population Size")
plt.ylabel("Final Best Fitness")
plt.title("Final Fitness Distribution by Population Size")
plt.grid(axis="y")
plt.tight_layout()
plt.show()
