import json
import glob
import numpy as np
import matplotlib.pyplot as plt

json_files = glob.glob("C:\\Users\\Todor\\Desktop\\results\\results *.json")

all_results = []
for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        all_results.append(data)

grouped = {}
for result in all_results:
    pop = result["population_size"]
    grouped.setdefault(pop, []).append(result)

plt.figure(figsize=(12, 6))
cutoff = 500

for pop_size, results in grouped.items():
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

    generations = np.arange(cutoff)
    plt.plot(generations, mean_prog, label=f"Pop {pop_size} (Mean)")
    plt.fill_between(generations, mean_prog - std_prog, mean_prog + std_prog, alpha=0.2)

plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Fitness Progression Grouped by Population Size (First 500 Generations)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
