import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_data(n_samples=100, seed=42):
    rng = np.random.default_rng(seed)

    x = np.arange(1, n_samples + 1, dtype=float)
    noise = rng.standard_normal(n_samples)
    y = 3 + 5 * x + noise

    return x, y


def load_data(filename):
    data = pd.read_csv(filename, header=None).to_numpy(dtype=float)

    X = data[:, :-1]
    y = data[:, -1]

    return X, y


def process_data(X, y, scale=False):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if scale:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1
        X = (X - mean) / std

    X = np.hstack((np.ones((X.shape[0], 1)), X))

    return X, y


def compute_cost(X, y, theta):
    m = len(y)

    predictions = X @ theta
    errors = predictions - y

    cost = (errors @ errors) / (2 * m)

    return cost


def gradient_descent(X, y, theta, learning_rate, n_iterations):
    m = len(y)
    cost_history = []

    for _ in range(n_iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (X.T @ errors) / m

        theta = theta - learning_rate * gradient

        cost_history.append(compute_cost(X, y, theta))

    return theta, np.array(cost_history)


def train(X, y, learning_rate=0.0001, n_iterations=50000):
    theta = np.zeros(X.shape[1])

    theta, cost_history = gradient_descent(
        X,
        y,
        theta,
        learning_rate,
        n_iterations
    )

    return theta, cost_history


def evaluate(X, y, theta):
    predictions = X @ theta
    errors = predictions - y

    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    cost = compute_cost(X, y, theta)

    return {
        "cost": cost,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }


def plot_training_error(cost_history, title, filename):
    plt.figure(figsize=(7, 5))

    plt.plot(
        range(1, len(cost_history) + 1),
        cost_history
    )

    plt.xlabel("Number of iterations")
    plt.ylabel("Training Error (Cost)")
    plt.title(title)

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        f"plots/{filename}.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


def plot_regression(X, y, theta, title, filename):
    x = X[:, 0]

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = theta[0] + theta[1] * x_line

    plt.figure(figsize=(7, 5))

    plt.scatter(
        x,
        y,
        s=20,
        alpha=0.7,
        label="Data points"
    )

    plt.plot(
        x_line,
        y_line,
        linewidth=2,
        label="Learned regression line"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        f"plots/{filename}.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


def run_experiment(name, filename, learning_rate, n_iterations):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    X, y = load_data(filename)

    X_processed, y_processed = process_data(
        X,
        y,
        scale=False
    )

    theta, cost_history = train(
        X_processed,
        y_processed,
        learning_rate,
        n_iterations
    )

    metrics = evaluate(
        X_processed,
        y_processed,
        theta
    )

    print("\nLearned parameters:")
    print(f"theta_0 (intercept) = {theta[0]:.6f}")
    print(f"theta_1 (slope)     = {theta[1]:.6f}")

    print("\nEvaluation:")
    print(f"Cost  = {metrics['cost']:.6f}")
    print(f"MSE   = {metrics['mse']:.6f}")
    print(f"RMSE  = {metrics['rmse']:.6f}")
    print(f"R²    = {metrics['r2']:.6f}")

    print(f"\nLearning rate = {learning_rate}")
    print(f"Iterations    = {n_iterations}")

    prefix = name.lower().replace(" ", "_")

    plot_training_error(
        cost_history,
        f"{name} - Training Error",
        f"{prefix}_training_error"
    )

    plot_regression(
        X,
        y,
        theta,
        f"{name} - Linear Regression",
        f"{prefix}_regression"
    )

    return theta


def main():
    os.makedirs("plots", exist_ok=True)

    x, y = generate_data(
        n_samples=100,
        seed=42
    )

    pd.DataFrame({
        "x": x,
        "y": y
    }).to_csv(
        "lab01_data.csv",
        index=False,
        header=False
    )

    print("Synthetic data saved to lab01_data.csv")

    run_experiment(
        name="Synthetic Data",
        filename="data_01.csv",
        learning_rate=0.0001,
        n_iterations=50000
    )

    run_experiment(
        name="Real Data",
        filename="data_01.csv",
        learning_rate=0.0001,
        n_iterations=50000
    )


if __name__ == "__main__":
    main()