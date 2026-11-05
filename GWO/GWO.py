
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# 1. Load Red Wine Quality Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

data = pd.read_csv(url, sep=";")

X = data.drop("quality", axis=1).values

# Convert quality into binary classification
y = (data["quality"] >= 6).astype(int).values

feature_names = data.drop("quality", axis=1).columns
total_features = X.shape[1]


# 2. Split Dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


# 3. Standardize Data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# 4. Fitness Function
def fitness(solution):

    selected = np.where(solution == 1)[0]

    if len(selected) == 0:
        return 1.0

    model = SVC(kernel="rbf")

    model.fit(X_train[:, selected], y_train)

    predictions = model.predict(X_val[:, selected])

    error = 1 - accuracy_score(y_val, predictions)

    feature_ratio = len(selected) / total_features

    alpha = 0.99
    beta = 0.01

    return alpha * error + beta * feature_ratio


# 5. Binary Conversion
def binary_conversion(position):

    probability = 1 / (1 + np.exp(-np.clip(position, -20, 20)))

    binary = (np.random.rand(len(position)) < probability).astype(int)

    if np.sum(binary) == 0:
        binary[np.argmax(probability)] = 1

    return binary


# 6. BGWO with Simulated Annealing
def BGWO_SA(population_size=10, max_iterations=20):

    dim = total_features

    positions = np.random.uniform(-2, 2, (population_size, dim))

    wolves = np.array([
        binary_conversion(position) for position in positions
    ])

    fitness_values = np.array([
        fitness(wolf) for wolf in wolves
    ])

    sorted_indices = np.argsort(fitness_values)

    alpha = wolves[sorted_indices[0]].copy()
    beta = wolves[sorted_indices[1]].copy()
    delta = wolves[sorted_indices[2]].copy()

    alpha_position = positions[sorted_indices[0]].copy()
    beta_position = positions[sorted_indices[1]].copy()
    delta_position = positions[sorted_indices[2]].copy()

    best_fitness = fitness_values[sorted_indices[0]]

    best_solution = alpha.copy()

    history = []

    temperature = 1.0

    for iteration in range(max_iterations):

        a = 2 - 2 * iteration / max_iterations

        for i in range(population_size):

            current_position = positions[i].copy()

            A1 = 2 * a * np.random.rand(dim) - a
            C1 = 2 * np.random.rand(dim)

            A2 = 2 * a * np.random.rand(dim) - a
            C2 = 2 * np.random.rand(dim)

            A3 = 2 * a * np.random.rand(dim) - a
            C3 = 2 * np.random.rand(dim)

            D_alpha = abs(
                C1 * alpha_position - current_position
            )

            D_beta = abs(
                C2 * beta_position - current_position
            )

            D_delta = abs(
                C3 * delta_position - current_position
            )

            X1 = alpha_position - A1 * D_alpha
            X2 = beta_position - A2 * D_beta
            X3 = delta_position - A3 * D_delta

            new_position = (X1 + X2 + X3) / 3

            new_solution = binary_conversion(new_position)

            new_fitness = fitness(new_solution)

            current_fitness = fitness(wolves[i])

            difference = new_fitness - current_fitness

            # Simulated Annealing Acceptance
            if difference < 0:

                wolves[i] = new_solution
                positions[i] = new_position

            else:

                probability = np.exp(
                    -difference / max(temperature, 1e-8)
                )

                if np.random.rand() < probability:

                    wolves[i] = new_solution
                    positions[i] = new_position

            # Update best solution
            candidate_fitness = fitness(wolves[i])

            if candidate_fitness < best_fitness:

                best_fitness = candidate_fitness
                best_solution = wolves[i].copy()

        # Rank wolves
        fitness_values = np.array([
            fitness(wolf) for wolf in wolves
        ])

        sorted_indices = np.argsort(fitness_values)

        alpha = wolves[sorted_indices[0]].copy()
        beta = wolves[sorted_indices[1]].copy()
        delta = wolves[sorted_indices[2]].copy()

        alpha_position = positions[sorted_indices[0]].copy()
        beta_position = positions[sorted_indices[1]].copy()
        delta_position = positions[sorted_indices[2]].copy()

        if fitness_values[sorted_indices[0]] < best_fitness:

            best_fitness = fitness_values[sorted_indices[0]]
            best_solution = alpha.copy()

        history.append(best_fitness)

        temperature *= 0.95

        print(
            "Iteration:", iteration + 1,
            "| Best Fitness:", round(best_fitness, 4),
            "| Selected Features:", np.sum(best_solution)
        )

    return best_solution, history


# 7. Run Algorithm
best_solution, history = BGWO_SA()

selected_features = np.where(best_solution == 1)[0]

print("\nSelected Features:")
print(feature_names[selected_features].tolist())

print("\nTotal Features:", total_features)
print("Selected Features:", len(selected_features))


# 8. Original Dataset Accuracy
original_model = SVC(kernel="rbf")

original_model.fit(X_train, y_train)

original_predictions = original_model.predict(X_test)

original_accuracy = accuracy_score(
    y_test, original_predictions
)


# 9. Selected Features Accuracy
selected_model = SVC(kernel="rbf")

selected_model.fit(
    X_train[:, selected_features], y_train
)

selected_predictions = selected_model.predict(
    X_test[:, selected_features]
)

selected_accuracy = accuracy_score(
    y_test, selected_predictions
)

print("\nOriginal Accuracy:",
      round(original_accuracy * 100, 2), "%")

print("Selected Features Accuracy:",
      round(selected_accuracy * 100, 2), "%")

print("Feature Reduction:",
      round((1 - len(selected_features) / total_features) * 100, 2),
      "%")


# 10. Fitness Graph
plt.figure(figsize=(8, 5))

plt.plot(history, marker="o")

plt.title("BGWO-SA Feature Selection")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")

plt.grid()
plt.show()


# 11. Accuracy Comparison
plt.figure(figsize=(7, 5))

plt.bar(
    ["All Features", "Selected Features"],
    [original_accuracy * 100, selected_accuracy * 100]
)

plt.title("Wine Quality Classification Accuracy")
plt.ylabel("Accuracy (%)")

plt.show()


# 12. Feature Reduction Graph
plt.figure(figsize=(7, 5))

plt.bar(
    ["Original Features", "Selected Features"],
    [total_features, len(selected_features)]
)

plt.title("Feature Reduction using BGWO-SA")
plt.ylabel("Number of Features")

plt.show()
