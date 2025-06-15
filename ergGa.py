from math import dist
from copy import deepcopy
import pygad
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

# — Optional: ensure any numpy-array printing uses max 3 decimals —
np.set_printoptions(precision=3, suppress=True)

# Data
nests = [
    {"id":1, "wasps":100, "x":25, "y":65},
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
    dist((a["x"], a["y"]), (b["x"], b["y"]))
    for a in nests for b in nests
)

def kill_function(n, d, dmax):
    return n * dmax / (20 * d + 1e-5)

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

def fitness_func(ga_instance, solution, solution_idx):
    bombs = [
        {"x": solution[0], "y": solution[1]},
        {"x": solution[2], "y": solution[3]},
        {"x": solution[4], "y": solution[5]}
    ]
    score = evaluate_solution(bombs, nests, dmax)
    min_bomb_dist = min(
        dist((bombs[i]["x"], bombs[i]["y"]), (bombs[j]["x"], bombs[j]["y"]))
        for i in range(3) for j in range(i + 1, 3)
    )
    penalty = 100 * max(0, 10 - min_bomb_dist)
    return score - penalty

best_solutions = []
best_fitnesses = []

def local_search(solution, step=2):
    best_solution = solution.copy()
    best_score = fitness_func(None, best_solution, 0)
    for i in range(len(solution)):
        for delta in (-step, step):
            temp = solution.copy()
            temp[i] = np.clip(temp[i] + delta, 0, 100)
            score = fitness_func(None, temp, 0)
            if score > best_score:
                best_solution, best_score = temp.copy(), score
    return best_solution, best_score

def on_generation(ga):
    sol, fit, _ = ga.best_solution()
    improved_sol, improved_fit = local_search(sol)
    if improved_fit > fit:
        ga.population[0] = improved_sol
        sol, fit = improved_sol, improved_fit
    best_solutions.append(sol)
    best_fitnesses.append(fit)
    print(f"Generation {ga.generations_completed}: Best Fitness = {fit:.3f}")

# GA setup
gene_space = {"low": 0, "high": 100}
ga_instance = pygad.GA(
    num_generations=350,
    num_parents_mating=20,
    fitness_func=fitness_func,
    sol_per_pop=100,
    num_genes=6,
    gene_space=gene_space,
    parent_selection_type="sss",
    keep_parents=2,
    crossover_type="two_points",
    mutation_type="random",
    mutation_percent_genes=20,
    on_generation=on_generation
)

# Run
ga_instance.run()

# Results
solution, solution_fitness, _ = ga_instance.best_solution()
print("Best solution:", [round(x,3) for x in solution])
print(f"Wasp kills: {solution_fitness:.3f}")

# Helper for 3-decimal tick formatting
def format_axes(ax):
    fmt = mtick.FormatStrFormatter('%.3f')
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

# Plots
def plot_progress(fitnesses):
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(fitnesses)+1), fitnesses)
    plt.xlabel("Generation")
    plt.ylabel("Best Wasp Kills")
    plt.title("Progress of Best Solution per Generation")
    plt.grid(True)
    ax = plt.gca()
    format_axes(ax)
    plt.show()

def plot_solution(sol):
    bombs = [(sol[0], sol[1]), (sol[2], sol[3]), (sol[4], sol[5])]
    nest_coords = [(n['x'], n['y']) for n in nests]
    plt.figure(figsize=(8,8))
    plt.scatter(*zip(*nest_coords), color='orange', label='Nests', s=100)
    plt.scatter(*zip(*bombs), color='red', marker='X', label='Bombs', s=100)
    plt.xlim(0,100); plt.ylim(0,100); plt.grid(True); plt.legend()
    plt.title('Bomb and Nest Locations')
    ax = plt.gca()
    format_axes(ax)
    plt.show()

def plot_population(pop):
    nest_coords = [(n['x'], n['y']) for n in nests]
    plt.figure(figsize=(8,8))
    plt.scatter(*zip(*nest_coords), color='orange', label='Nests', s=100)
    for sol in pop:
        b = [(sol[0],sol[1]),(sol[2],sol[3]),(sol[4],sol[5])]
        plt.scatter(*zip(*b), color='blue', alpha=0.3, s=20)
    plt.xlim(0,100); plt.ylim(0,100); plt.grid(True); plt.legend(['Nests','Population Bombs'])
    plt.title('Population Bomb Positions in Last Generation')
    ax = plt.gca()
    format_axes(ax)
    plt.show()

# Show plots
plot_progress(best_fitnesses)
plot_solution(best_solutions[-1])
plot_population(ga_instance.population)
