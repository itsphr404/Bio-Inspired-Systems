import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# TRAVELLING SALESMAN PROBLEM USING ANT COLONY SYSTEM
# Inspired by Dorigo & Gambardella, IEEE, 1997
# Includes random-proportional rule visualization
# ============================================================

# -------------------- PARAMETERS ----------------------------

NUM_CITIES = 25
NUM_ANTS = 25
ITERATIONS = 200

ALPHA = 1.0
BETA = 3.0
RHO = 0.10
PHI = 0.10
Q0 = 0.90
Q = 1.0

SEED = 42
rng = np.random.default_rng(SEED)

# -------------------- CREATE CITIES -------------------------

cities = rng.uniform(0, 100, size=(NUM_CITIES, 2))

difference = cities[:, None, :] - cities[None, :, :]
distances = np.sqrt(np.sum(difference ** 2, axis=2))

heuristic = np.zeros_like(distances)
np.divide(
    1.0, distances,
    out=heuristic,
    where=distances > 0
)

initial_pheromone = 1.0 / (
    NUM_CITIES * np.mean(distances[distances > 0])
)

pheromone = np.full(
    (NUM_CITIES, NUM_CITIES),
    initial_pheromone
)

np.fill_diagonal(pheromone, 0.0)


# -------------------- TOUR DISTANCE -------------------------

def tour_length(tour):
    """Calculate total closed-tour distance."""
    route = np.asarray(tour)

    return float(
        np.sum(distances[route, np.roll(route, -1)])
    )


# -------------------- NEAREST NEIGHBOUR ---------------------

def nearest_neighbour(start):
    """Generate a greedy tour from a starting city."""
    unvisited = set(range(NUM_CITIES))
    unvisited.remove(start)

    tour = [start]
    current = start

    while unvisited:
        nxt = min(
            unvisited,
            key=lambda city: distances[current, city]
        )

        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    return tour


baseline_tours = [
    nearest_neighbour(start)
    for start in range(NUM_CITIES)
]

baseline_tour = min(
    baseline_tours,
    key=tour_length
)

baseline_distance = tour_length(baseline_tour)


# -------------------- 2-OPT LOCAL SEARCH --------------------

def two_opt(tour):
    """Improve a route by reversing route segments."""
    route = list(tour)
    improved = True

    while improved:
        improved = False
        n = len(route)

        for i in range(n - 1):
            for j in range(i + 2, n):

                if i == 0 and j == n - 1:
                    continue

                a = route[i]
                b = route[(i + 1) % n]
                c = route[j]
                d = route[(j + 1) % n]

                old_cost = (
                    distances[a, b] + distances[c, d]
                )

                new_cost = (
                    distances[a, c] + distances[b, d]
                )

                if new_cost < old_cost - 1e-12:
                    route[i + 1:j + 1] = reversed(
                        route[i + 1:j + 1]
                    )

                    improved = True
                    break

            if improved:
                break

    return route


# -------------------- ACS TOUR CONSTRUCTION -----------------

def construct_tour(start):
    """Construct a tour using the ACS selection rule."""
    global pheromone

    tour = [start]
    unvisited = set(range(NUM_CITIES))
    unvisited.remove(start)

    current = start

    while unvisited:
        candidates = np.array(sorted(unvisited))

        # Pheromone and distance-based desirability
        desirability = (
            pheromone[current, candidates] ** ALPHA
        ) * (
            heuristic[current, candidates] ** BETA
        )

        total = desirability.sum()

        # Normalize probabilities
        if total <= 0 or not np.isfinite(total):
            probabilities = np.full(
                len(candidates),
                1.0 / len(candidates)
            )
        else:
            probabilities = desirability / total

        # Pseudorandom proportional rule
        if rng.random() < Q0:
            nxt = candidates[np.argmax(desirability)]
        else:
            nxt = rng.choice(
                candidates,
                p=probabilities
            )

        # Local pheromone update
        new_tau = (
            (1 - PHI) * pheromone[current, nxt]
            + PHI * initial_pheromone
        )

        pheromone[current, nxt] = new_tau
        pheromone[nxt, current] = new_tau

        tour.append(int(nxt))
        unvisited.remove(int(nxt))
        current = int(nxt)

    return tour


# -------------------- MAIN ACO ALGORITHM --------------------

best_tour = None
best_distance = float("inf")

best_history = []
average_history = []

for iteration in range(ITERATIONS):

    ant_tours = []
    ant_lengths = []

    for ant in range(NUM_ANTS):

        # Spread starting cities across the population
        start = ant % NUM_CITIES

        tour = construct_tour(start)
        tour = two_opt(tour)

        length = tour_length(tour)

        ant_tours.append(tour)
        ant_lengths.append(length)

        if length < best_distance:
            best_distance = length
            best_tour = tour[:]

    # Record performance
    best_history.append(best_distance)
    average_history.append(np.mean(ant_lengths))

    # Global pheromone evaporation
    pheromone *= (1 - RHO)

    # Reinforce the global-best tour
    for i in range(NUM_CITIES):
        a = best_tour[i]
        b = best_tour[(i + 1) % NUM_CITIES]

        pheromone[a, b] += RHO * Q / best_distance
        pheromone[b, a] = pheromone[a, b]

    np.fill_diagonal(pheromone, 0.0)

    if iteration % 20 == 0 or iteration == ITERATIONS - 1:
        print(
            f"Iteration {iteration + 1:3d}/{ITERATIONS} | "
            f"Best distance: {best_distance:.2f} | "
            f"Average distance: {np.mean(ant_lengths):.2f}"
        )


# -------------------- FINAL RESULTS -------------------------

improvement = (
    (baseline_distance - best_distance)
    / baseline_distance * 100
)

print("\n========== FINAL RESULTS ==========")
print(f"Number of cities           : {NUM_CITIES}")
print(f"Number of ants             : {NUM_ANTS}")
print(f"Number of iterations       : {ITERATIONS}")
print(f"Nearest Neighbour distance : {baseline_distance:.2f}")
print(f"ACO best distance          : {best_distance:.2f}")
print(f"Improvement vs baseline    : {improvement:.2f}%")
print(f"Best tour                  : {best_tour}")
print(f"Return to starting city    : {best_tour[0]}")


# -------------------- SAVE CSV RESULTS ----------------------

results = pd.DataFrame({
    "Iteration": np.arange(1, ITERATIONS + 1),
    "Best_Distance": best_history,
    "Average_Ant_Distance": average_history
})

results.to_csv("aco_convergence.csv", index=False)

print("\nConvergence data saved to aco_convergence.csv")


# ============================================================
# RANDOM-PROPORTIONAL RULE NORMALIZATION
# ============================================================

# Example decision at city 0.
# Assume cities 0, 1 and 2 have already been visited.

current_city = 0
visited_cities = {0, 1, 2}

candidate_cities = np.array([
    city for city in range(NUM_CITIES)
    if city not in visited_cities
])

# Calculate desirability from the final pheromone matrix
desirability = (
    pheromone[current_city, candidate_cities] ** ALPHA
) * (
    heuristic[current_city, candidate_cities] ** BETA
)

total = desirability.sum()

if total > 0 and np.isfinite(total):
    probabilities = desirability / total
else:
    probabilities = np.full(
        len(candidate_cities),
        1.0 / len(candidate_cities)
    )

print("\n========== NORMALIZATION CHECK ==========")
print(f"Current city: {current_city}")
print("Candidate cities:", candidate_cities.tolist())
print(f"Sum of probabilities: {probabilities.sum():.6f}")

probability_table = pd.DataFrame({
    "Candidate_City": candidate_cities,
    "Desirability": desirability,
    "Normalized_Probability": probabilities
})

print(probability_table.to_string(index=False))

probability_table.to_csv(
    "aco_selection_probabilities.csv",
    index=False
)


# ============================================================
# VISUALIZATION: FOUR GRAPHS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# -------------------- GRAPH 1: BEST TOUR --------------------

ax = axes[0, 0]

closed_tour = best_tour + [best_tour[0]]
route_coords = cities[closed_tour]

ax.plot(
    route_coords[:, 0],
    route_coords[:, 1],
    "o-",
    color="royalblue",
    linewidth=1.5,
    markersize=5
)

ax.scatter(
    cities[:, 0],
    cities[:, 1],
    color="red",
    s=35,
    zorder=3
)

for i, (x, y) in enumerate(cities):
    ax.annotate(
        str(i),
        (x, y),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8
    )

ax.set_title(
    f"Best ACO Tour\nDistance = {best_distance:.2f}"
)

ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
ax.grid(True, alpha=0.3)


# -------------------- GRAPH 2: CONVERGENCE ------------------

ax = axes[0, 1]

ax.plot(
    range(1, ITERATIONS + 1),
    best_history,
    color="green",
    linewidth=2,
    label="Best-so-far distance"
)

ax.plot(
    range(1, ITERATIONS + 1),
    average_history,
    color="orange",
    alpha=0.8,
    label="Average ant distance"
)

ax.set_title("ACO Convergence")
ax.set_xlabel("Iteration")
ax.set_ylabel("Tour distance")
ax.legend()
ax.grid(True, alpha=0.3)


# -------------------- GRAPH 3: BASELINE COMPARISON -----------

ax = axes[1, 0]

labels = ["Nearest Neighbour", "ACO + 2-opt"]
values = [baseline_distance, best_distance]

bars = ax.bar(
    labels,
    values,
    color=["darkorange", "royalblue"]
)

ax.bar_label(bars, fmt="%.2f", padding=3)

ax.set_title("Tour Distance Comparison")
ax.set_ylabel("Total distance (lower is better)")
ax.grid(axis="y", alpha=0.3)


# -------------------- GRAPH 4: NORMALIZATION ---------------

ax = axes[1, 1]

ax.bar(
    candidate_cities,
    probabilities,
    color="teal",
    edgecolor="black"
)

ax.set_title(
    "Random-Proportional Rule\n"
    "Normalized Selection Probabilities"
)

ax.set_xlabel("Candidate City")
ax.set_ylabel("Selection Probability")
ax.set_xticks(candidate_cities)
ax.tick_params(axis="x", labelsize=8)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "aco_tsp_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nAll graphs saved to aco_tsp_results.png")
print("Probability data saved to aco_selection_probabilities.csv")