from math import dist
from copy import deepcopy
import pygad
import matplotlib.pyplot as plt

nests = [
    {"id":1, "wasps":100, "x":25, "y": 65},
    {"id":2, "wasps":200, "x":23, "y":8},
    {"id":3, "wasps":327, "x":7,  "y":13},
    {"id":4, "wasps":440, "x":95, "y":53},
    {"id":5, "wasps":450, "x":3,  "y":3},
    {"id":6, "wasps":639, "x":54, "y":56},
    {"id":7, "wasps":650, "x":67, "y":78},
    {"id":8, "wasps":678, "x":32, "y":4},
    {"id":9, "wasps":750, "x":24, "y":76},
    {"id":10,"wasps":801, "x":66, "y":89},
    {"id":11,"wasps":945, "x":84, "y":4},
    {"id":12,"wasps":967, "x":34, "y":23}
]

dmax = max(
    dist((a["x"], a["y"]), (b["x"], b["y"])) for a in nests for b in nests
)

def kill_function(n, d, dmax):
    return n * dmax / (20 * d + 0.00001)

def evaluate_solution(bombs, nests, dmax):
    nests_copy = deepcopy(nests)
    total_kills = 0

    for bomb in bombs:
        for nest in nests_copy:
            if nest["wasps"] == 0:
                continue
            d = dist((bomb["x"], bomb["y"]), (nest["x"], nest["y"]))
            kills = min(nest["wasps"], kill_function(nest["wasps"], d, dmax))
            total_kills += kills
            nest["wasps"] -= kills

    return total_kills

test_bombs = [
    {"x": 20, "y": 20},
    {"x": 60, "y": 60},
    {"x": 80, "y": 10}
]

def fitness_func(ga_instance, solution, solution_idx):
    bombs = [
        {"x": solution[0], "y": solution[1]},
        {"x": solution[2], "y": solution[3]},
        {"x": solution[4], "y": solution[5]}
    ]
    score = evaluate_solution(bombs, nests, dmax)
    return score

# --- PY-GAD SETUP ---
gene_space = {"low": 0, "high": 100}

ga_instance = pygad.GA(
    num_generations=350,
    num_parents_mating=20,
    fitness_func=fitness_func,
    sol_per_pop=60,
    num_genes=6,
    gene_space=gene_space,
    parent_selection_type="sss",
    keep_parents=2,
    crossover_type="two_points",
    mutation_type="random",
    mutation_percent_genes=20
)

# --- RUN GA ---
ga_instance.run()

# --- RESULTS ---
solution, solution_fitness, _ = ga_instance.best_solution()
print("Best solution:", solution)
print("Wasp kills:", solution_fitness)

def plot_solution(solution):
    bombs = [(solution[0], solution[1]), (solution[2], solution[3]), (solution[4], solution[5])]
    nest_coords = [(n['x'], n['y']) for n in nests]

    plt.figure(figsize=(8, 8))
    plt.scatter(*zip(*nest_coords), color='orange', label='Nests', s=100)
    plt.scatter(*zip(*bombs), color='red', marker='X', label='Bombs', s=100)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.title('Bomb and Nest Locations')
    plt.show()

plot_solution(solution)