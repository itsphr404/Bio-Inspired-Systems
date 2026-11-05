import random
from collections import Counter

# ---------------------------------------
# LABOUR SHIFT SCHEDULING USING GA
# ---------------------------------------

DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

SHIFTS = ["Morning", "Evening", "Night"]

WORKERS = [
    "Worker_1", "Worker_2", "Worker_3", "Worker_4",
    "Worker_5", "Worker_6", "Worker_7", "Worker_8",
    "Worker_9", "Worker_10", "Worker_11", "Worker_12"
]

# Required workers for each shift
REQUIRED_WORKERS = 2

# Maximum shifts a worker can work in a week
MAX_SHIFTS_PER_WORKER = 5

POPULATION_SIZE = 100
GENERATIONS = 100
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.4
ELITE_COUNT = 2

random.seed(42)

# A gene represents one worker assigned to one shift.
SLOTS = [
    (day, shift)
    for day in DAYS
    for shift in SHIFTS
    for _ in range(REQUIRED_WORKERS)
]

# Example unavailability:
# Worker_1 cannot work Monday morning.
UNAVAILABLE = {
    ("Worker_1", "Monday", "Morning"),
    ("Worker_3", "Friday", "Night"),
    ("Worker_5", "Sunday", "Evening"),
}


def is_valid(worker, slot):
    day, shift = slot
    return (worker, day, shift) not in UNAVAILABLE


def create_individual():
    """Create a candidate weekly schedule."""
    schedule = []

    for slot in SLOTS:
        candidates = [
            w for w in WORKERS
            if is_valid(w, slot)
        ]
        schedule.append(random.choice(candidates))

    return schedule


def fitness(schedule):
    """Higher fitness means a better schedule."""
    score = 1000
    assignments = Counter()
    daily_assignments = Counter()
    worker_counts = Counter()

    # Check staffing and worker assignments.
    for i, worker in enumerate(schedule):
        day, shift = SLOTS[i]

        assignments[(day, shift)] += 1
        daily_assignments[(worker, day)] += 1
        worker_counts[worker] += 1

        if not is_valid(worker, (day, shift)):
            score -= 200

    # Avoid assigning one worker multiple times
    # to the same shift.
    for count in assignments.values():
        if count > REQUIRED_WORKERS:
            score -= 100 * (count - REQUIRED_WORKERS)

    # Penalize workers scheduled for multiple shifts
    # on the same day.
    for count in daily_assignments.values():
        if count > 1:
            score -= 100 * (count - 1)

    # Penalize weekly workload violations.
    for worker in WORKERS:
        if worker_counts[worker] > MAX_SHIFTS_PER_WORKER:
            score -= 50 * (
                worker_counts[worker] - MAX_SHIFTS_PER_WORKER
            )

    # Encourage balanced workloads.
    average = len(schedule) / len(WORKERS)

    for worker in WORKERS:
        score -= 3 * abs(worker_counts[worker] - average)

    return score


def tournament_selection(population, k=3):
    """Select the best individual from a random group."""
    candidates = random.sample(population, k)
    return max(candidates, key=fitness)[:]


def one_point_crossover(parent1, parent2):
    """Combine two schedules at one cut point."""
    if random.random() > CROSSOVER_RATE:
        return parent1[:], parent2[:]

    point = random.randint(1, len(SLOTS) - 1)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2


def insertion_mutation(schedule):
    """Move one gene to another position."""
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(schedule)), 2)
        gene = schedule.pop(i)
        schedule.insert(j, gene)

    return schedule


def generate_schedule():
    population = [
        create_individual()
        for _ in range(POPULATION_SIZE)
    ]

    best = None

    for generation in range(GENERATIONS):
        population.sort(key=fitness, reverse=True)

        if best is None or fitness(population[0]) > fitness(best):
            best = population[0][:]

        if generation % 10 == 0:
            print(
                f"Generation {generation}: "
                f"Fitness = {fitness(population[0])}"
            )

        new_population = [
            individual[:]
            for individual in population[:ELITE_COUNT]
        ]

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            child1, child2 = one_point_crossover(
                parent1, parent2
            )

            new_population.append(
                insertion_mutation(child1)
            )

            if len(new_population) < POPULATION_SIZE:
                new_population.append(
                    insertion_mutation(child2)
                )

        population = new_population

    population.sort(key=fitness, reverse=True)

    if fitness(population[0]) > fitness(best):
        best = population[0][:]

    return best


def display_schedule(schedule):
    print("\n========== LABOUR SHIFT SCHEDULE ==========")

    for day in DAYS:
        print(f"\n{day}")

        for shift in SHIFTS:
            workers = [
                schedule[i]
                for i, slot in enumerate(SLOTS)
                if slot == (day, shift)
            ]

            print(f"  {shift:8}: {', '.join(workers)}")

    print("\n========== WORKER SUMMARY ==========")

    counts = Counter(schedule)

    for worker in WORKERS:
        print(f"{worker}: {counts[worker]} shifts")

    print("\nFinal fitness:", fitness(schedule))


if __name__ == "__main__":
    best_schedule = generate_schedule()
    display_schedule(best_schedule)