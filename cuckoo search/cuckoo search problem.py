
"""
Phishing Detection Using Deep Learning and Cuckoo Search
Dataset: UCI Phishing Websites (ID 327)

Outputs:
  1. performance_comparison.png
  2. confusion_matrix.png
  3. cuckoo_optimization.png

Note: This is an independent implementation inspired by the
IEEE paper's topic, not an exact reproduction of its model.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

print("Downloading UCI Phishing Websites dataset...")

dataset = fetch_ucirepo(id=327)
X = dataset.data.features.copy()
target = dataset.data.targets.iloc[:, 0]

# UCI dataset convention: -1 = phishing, 1 = legitimate.
# Convert phishing to class 1 and legitimate to class 0.
target = pd.to_numeric(target)
y = (target == -1).astype(int).to_numpy()

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median()).to_numpy(dtype=float)

print("Dataset shape:", X.shape)
print("Legitimate websites:", int(np.sum(y == 0)))
print("Phishing websites:", int(np.sum(y == 1)))


# ---------------------------------------------------------
# 2. SPLIT DATA: TEST SET IS KEPT SEPARATE
# ---------------------------------------------------------

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_trainval
)

# Fit the scaler only on the training data to avoid leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nTraining records:", len(y_train))
print("Validation records:", len(y_val))
print("Held-out test records:", len(y_test))


# ---------------------------------------------------------
# 3. BASELINE DEEP-LEARNING MODEL
# ---------------------------------------------------------

print("\nTraining baseline MLP...")

baseline = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=12,
    random_state=RANDOM_STATE
)

baseline.fit(X_train_scaled, y_train)
baseline_pred = baseline.predict(X_test_scaled)


# ---------------------------------------------------------
# 4. CUCKOO SEARCH OPTIMIZER
# ---------------------------------------------------------
# Each nest represents three hyperparameters:
# x[0]: hidden-layer width, 32 to 128
# x[1]: log10(alpha), -5 to -2
# x[2]: log10(learning rate), -4 to -2
#
# Fitness = validation F1-score.
# Higher is better.


def decode_nest(position):
    """Convert a [0, 1]^3 vector into MLP hyperparameters."""
    hidden = int(round(32 + position[0] * 96))
    alpha = 10 ** (-5 + position[1] * 3)
    learning_rate = 10 ** (-4 + position[2] * 2)

    return hidden, alpha, learning_rate


def evaluate_nest(position):
    """Train one MLP and return its validation F1-score."""
    hidden, alpha, learning_rate = decode_nest(position)

    model = MLPClassifier(
        hidden_layer_sizes=(hidden, max(8, hidden // 2)),
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate,
        max_iter=120,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=RANDOM_STATE
    )

    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_val_scaled)

    return f1_score(
        y_val, predictions,
        pos_label=1,
        zero_division=0
    )


def levy_flight(dimension=3, beta=1.5):
    """Generate a Levy-flight step for Cuckoo Search."""
    from scipy.special import gamma

    sigma = (
        gamma(1 + beta) * np.sin(np.pi * beta / 2)
        / (
            gamma((1 + beta) / 2)
            * beta
            * 2 ** ((beta - 1) / 2)
        )
    ) ** (1 / beta)

    u = rng.normal(0, sigma, dimension)
    v = rng.normal(0, 1, dimension)

    return u / (np.abs(v) ** (1 / beta) + 1e-12)


def cuckoo_search(n_nests=6, generations=5, pa=0.25):
    """
    Cuckoo Search:
      1. Initialize nests.
      2. Generate new candidates using Levy flights.
      3. Keep candidates with better fitness.
      4. Abandon a fraction of nests.
      5. Track the best solution.
    """
    nests = rng.uniform(0, 1, size=(n_nests, 3))
    scores = np.array([evaluate_nest(n) for n in nests])

    best_idx = np.argmax(scores)
    best_nest = nests[best_idx].copy()
    best_score = float(scores[best_idx])
    history = [best_score]

    print("\nStarting Cuckoo Search...")
    print(f"Initial best validation F1: {best_score:.4f}")

    for generation in range(generations):
        for i in range(n_nests):
            # Levy-flight exploration around the current nest.
            step = levy_flight()
            candidate = nests[i] + 0.15 * step * (
                nests[i] - best_nest + rng.normal(size=3) * 0.1
            )
            candidate = np.clip(candidate, 0, 1)

            candidate_score = evaluate_nest(candidate)

            # Retain the candidate if it improves the nest.
            j = rng.integers(0, n_nests)
            if candidate_score > scores[j]:
                nests[j] = candidate
                scores[j] = candidate_score

            if candidate_score > best_score:
                best_nest = candidate.copy()
                best_score = float(candidate_score)

        # Abandon some nests and replace them with new candidates.
        abandon_count = max(1, int(pa * n_nests))
        worst_indices = np.argsort(scores)[:abandon_count]

        for idx in worst_indices:
            nests[idx] = rng.uniform(0, 1, size=3)
            scores[idx] = evaluate_nest(nests[idx])

            if scores[idx] > best_score:
                best_nest = nests[idx].copy()
                best_score = float(scores[idx])

        history.append(best_score)

        print(
            f"Generation {generation + 1}/{generations}: "
            f"best validation F1 = {best_score:.4f}"
        )

    return best_nest, best_score, history


best_position, best_val_f1, history = cuckoo_search()

best_hidden, best_alpha, best_lr = decode_nest(best_position)

print("\nBest hyperparameters:")
print("Hidden-layer width:", best_hidden)
print("Alpha (regularization):", best_alpha)
print("Learning rate:", best_lr)
print("Best validation F1:", round(best_val_f1, 4))


# ---------------------------------------------------------
# 5. TRAIN THE OPTIMIZED MODEL
# ---------------------------------------------------------

print("\nTraining final optimized model...")

# Retrain using training + validation data.
X_full_train = np.vstack([X_train, X_val])
y_full_train = np.concatenate([y_train, y_val])

final_scaler = StandardScaler()
X_full_scaled = final_scaler.fit_transform(X_full_train)
X_test_final = final_scaler.transform(X_test)

optimized = MLPClassifier(
    hidden_layer_sizes=(
        best_hidden, max(8, best_hidden // 2)
    ),
    activation="relu",
    solver="adam",
    alpha=best_alpha,
    learning_rate_init=best_lr,
    max_iter=250,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    random_state=RANDOM_STATE
)

optimized.fit(X_full_scaled, y_full_train)
optimized_pred = optimized.predict(X_test_final)


# ---------------------------------------------------------
# 6. EVALUATION
# ---------------------------------------------------------

def calculate_metrics(actual, predicted):
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "Recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "F1-score": f1_score(
            actual, predicted, zero_division=0
        )
    }


baseline_metrics = calculate_metrics(y_test, baseline_pred)
optimized_metrics = calculate_metrics(y_test, optimized_pred)

results = pd.DataFrame(
    [baseline_metrics, optimized_metrics],
    index=["Baseline MLP", "Cuckoo-optimized MLP"]
)

print("\nTEST SET PERFORMANCE")
print((results * 100).round(2).to_string())

print("\nOptimized model classification report:")
print(
    classification_report(
        y_test,
        optimized_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    )
)

results.to_csv("performance_results.csv")


# ---------------------------------------------------------
# 7. GRAPH 1: PERFORMANCE COMPARISON
# ---------------------------------------------------------

ax = (results * 100).plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 100),
    rot=0
)

ax.set_title("Baseline MLP vs Cuckoo-Optimized MLP")
ax.set_ylabel("Score (%)")
ax.set_xlabel("Model")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("performance_comparison.png", dpi=300)
plt.show()


# ---------------------------------------------------------
# 8. GRAPH 2: CONFUSION MATRIX
# ---------------------------------------------------------

cm = confusion_matrix(
    y_test, optimized_pred, labels=[0, 1]
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Phishing"],
    yticklabels=["Legitimate", "Phishing"]
)
plt.title("Cuckoo-Optimized MLP: Confusion Matrix")
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


# ---------------------------------------------------------
# 9. GRAPH 3: CUCKOO SEARCH CONVERGENCE
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(
    range(len(history)),
    history,
    marker="o",
    color="darkgreen"
)
plt.title("Cuckoo Search Optimization Progress")
plt.xlabel("Generation")
plt.ylabel("Best validation F1-score")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cuckoo_optimization.png", dpi=300)
plt.show()


# ---------------------------------------------------------
# 10. SAVE TEST PREDICTIONS
# ---------------------------------------------------------

prediction_output = pd.DataFrame({
    "Actual": y_test,
    "Predicted": optimized_pred,
    "Actual_Label": np.where(y_test == 1, "Phishing", "Legitimate"),
    "Predicted_Label": np.where(
        optimized_pred == 1, "Phishing", "Legitimate"
    )
})

prediction_output.to_csv("test_predictions.csv", index=False)

print("\nFiles generated:")
print("- performance_results.csv")
print("- performance_comparison.png")
print("- confusion_matrix.png")
print("- cuckoo_optimization.png")
print("- test_predictions.csv")
print("\nExperiment completed.")
