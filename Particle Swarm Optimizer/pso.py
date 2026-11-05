
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PSO FOR ROBOT PATH PLANNING
# ==========================================

# 1. Define the Problem
START = np.array([0.0, 0.0])
GOAL = np.array([10.0, 10.0])

# Circular obstacles: (x, y, radius)
OBSTACLES = [
    (3.0, 3.0, 1.2),
    (5.0, 5.0, 1.5),
    (7.0, 7.0, 1.2),
    (3.0, 7.0, 1.0),
    (7.0, 3.0, 1.0)
]

# 2. Initialize Parameters
NUM_PARTICLES = 50
NUM_WAYPOINTS = 5
MAX_ITERATIONS = 200

W = 0.7
C1 = 1.5
C2 = 1.5

DIMENSIONS = NUM_WAYPOINTS * 2
LOWER_BOUND = 0
UPPER_BOUND = 10

COLLISION_PENALTY = 1000


# ==========================================
# 3. Fitness Function
# ==========================================

def fitness(position):

    waypoints = position.reshape(NUM_WAYPOINTS, 2)

    path = np.vstack((START, waypoints, GOAL))

    # Calculate total path length
    path_length = 0

    for i in range(len(path) - 1):

        distance = np.linalg.norm(path[i + 1] - path[i])

        path_length += distance

    # Calculate collision penalty
    penalty = 0

    for i in range(len(path) - 1):

        p1 = path[i]
        p2 = path[i + 1]

        segment = p2 - p1
        denominator = np.dot(segment, segment)

        for ox, oy, radius in OBSTACLES:

            center = np.array([ox, oy])

            if denominator == 0:

                closest = p1

            else:

                t = np.dot(center - p1, segment) / denominator

                t = np.clip(t, 0, 1)

                closest = p1 + t * segment

            distance = np.linalg.norm(center - closest)

            if distance < radius:

                penalty += COLLISION_PENALTY * (
                    radius - distance + 0.1
                )

    return path_length + penalty


# ==========================================
# 4. Initialize Particles
# ==========================================

positions = np.random.uniform(
    LOWER_BOUND,
    UPPER_BOUND,
    (NUM_PARTICLES, DIMENSIONS)
)

velocities = np.random.uniform(
    -1, 1,
    (NUM_PARTICLES, DIMENSIONS)
)

personal_best_positions = positions.copy()

personal_best_fitness = np.array([
    fitness(p) for p in positions
])

best_index = np.argmin(personal_best_fitness)

global_best_position = personal_best_positions[
    best_index
].copy()

global_best_fitness = personal_best_fitness[best_index]

convergence = []


# ==========================================
# 5. PSO Optimization
# ==========================================

for iteration in range(MAX_ITERATIONS):

    for i in range(NUM_PARTICLES):

        current_fitness = fitness(positions[i])

        # Update Personal Best
        if current_fitness < personal_best_fitness[i]:

            personal_best_fitness[i] = current_fitness

            personal_best_positions[i] = positions[i].copy()

        # Update Global Best
        if personal_best_fitness[i] < global_best_fitness:

            global_best_fitness = personal_best_fitness[i]

            global_best_position = (
                personal_best_positions[i].copy()
            )

        # Generate random values
        r1 = np.random.rand(DIMENSIONS)
        r2 = np.random.rand(DIMENSIONS)

        # Update Velocity
        velocities[i] = (
            W * velocities[i]
            + C1 * r1 * (
                personal_best_positions[i] - positions[i]
            )
            + C2 * r2 * (
                global_best_position - positions[i]
            )
        )

        # Update Position
        positions[i] = positions[i] + velocities[i]

        # Apply boundary constraints
        positions[i] = np.clip(
            positions[i],
            LOWER_BOUND,
            UPPER_BOUND
        )

    convergence.append(global_best_fitness)


# ==========================================
# 6. Output Best Solution
# ==========================================

best_waypoints = global_best_position.reshape(
    NUM_WAYPOINTS, 2
)

optimized_path = np.vstack((
    START,
    best_waypoints,
    GOAL
))

path_length = np.sum(
    np.linalg.norm(
        np.diff(optimized_path, axis=0),
        axis=1
    )
)

print("\nPSO ROBOT PATH PLANNING RESULTS")
print("--------------------------------")

print("Starting Point:", START)

print("Goal Point:", GOAL)

print("Best Fitness:", round(global_best_fitness, 3))

print("Optimized Path Length:", round(path_length, 3))

print("\nOptimized Waypoints:")

for i, point in enumerate(best_waypoints, 1):

    print(
        f"Waypoint {i}: "
        f"({point[0]:.2f}, {point[1]:.2f})"
    )


# ==========================================
# 7. Plot Robot Path
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Robot Path Graph
ax = axes[0]

# Draw obstacles
for ox, oy, radius in OBSTACLES:

    obstacle = plt.Circle(
        (ox, oy),
        radius,
        color="red",
        alpha=0.35
    )

    ax.add_patch(obstacle)

    ax.plot(ox, oy, "r+")

# Draw optimized path
ax.plot(
    optimized_path[:, 0],
    optimized_path[:, 1],
    "b-o",
    linewidth=2,
    label="Optimized Path"
)

# Start and Goal
ax.scatter(
    START[0], START[1],
    color="green",
    s=100,
    label="Start"
)

ax.scatter(
    GOAL[0], GOAL[1],
    color="purple",
    s=100,
    label="Goal"
)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")

ax.set_title("PSO Robot Path Planning")

ax.grid(True)
ax.legend()
ax.set_aspect("equal")


# ==========================================
# 8. Plot Convergence Curve
# ==========================================

axes[1].plot(
    range(1, MAX_ITERATIONS + 1),
    convergence,
    linewidth=2
)

axes[1].set_xlabel("Iteration")

axes[1].set_ylabel("Best Fitness")

axes[1].set_title("PSO Convergence Curve")

axes[1].grid(True)

plt.tight_layout()

plt.show()
