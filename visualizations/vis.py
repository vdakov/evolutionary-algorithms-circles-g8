import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

json_files = glob.glob("C:\\Users\\Todor\\Desktop\\results\\results *.json")

all_results = []
for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        if data["population_size"] == 100:
            all_results.append(data)

grouped = {}
for result in all_results:
    init = result["initialization_strategy"]
    grouped.setdefault(init, []).append(result)

plt.figure(figsize=(12, 6))
cutoff = 200

for init_strategy, results in grouped.items():
    all_progressions = []

    for r in results:
        for run in r["runs"]:
            prog = np.array(run["progression"][:cutoff])
            if len(prog) < cutoff:
                prog = np.pad(prog, (0, cutoff - len(prog)), constant_values=prog[-1])
            all_progressions.append(prog)

    all_progressions = np.array(all_progressions)
    mean_prog = np.mean(all_progressions, axis=0)
    std_prog = np.std(all_progressions, axis=0)
    best_run = all_progressions[np.argmax(all_progressions[:, -1])]

    generations = np.arange(cutoff)
    plt.plot(generations, mean_prog, label=f"{init_strategy} (Mean)")
    plt.plot(generations, best_run, linestyle="--", label=f"{init_strategy} (Best Run)")
    plt.fill_between(generations, mean_prog - std_prog, mean_prog + std_prog, alpha=0.2)

plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title(
    "Progression by Initialization Strategy (Population = 100, First 500 Generations)"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
