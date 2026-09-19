"""
CSE-4155 Introduction to Machine Learning Lab
Lab 1: Linear Regression with One Variable

Implements linear regression from scratch using gradient descent.

Expected CSV format:
x,y
1,...
2,...
...
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filename="dataset.csv"):
    """
    Load x and y values from a CSV file.

    Parameters
    ----------
    filename : str
        Path to the CSV file.

    Returns
    -------
    x : np.ndarray
        Feature values.
    y : np.ndarray
        Target values.
    """
    data = pd.read_csv(filename)

    if "x" not in data.columns or "y" not in data.columns:
        raise ValueError("CSV file must contain 'x' and 'y' columns.")

    x = data["x"].to_numpy(dtype=float)
    y = data["y"].to_numpy(dtype=float)

    return x, y


def process_data(x, y, scale=True):
    """
    Prepare the feature matrix.

    A dummy feature x0 = 1 is added so that:
        X = [1, x]

    Optional standardization is applied to x before adding
    the dummy feature.

    Returns
    -------
    X : np.ndarray
        Processed feature matrix.
    y : np.ndarray
        Target vector.
    scaler : dict
        Mean and standard deviation used for feature scaling.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    scaler = {
        "scaled": scale,
        "mean": 0.0,
        "std": 1.0
    }

    if scale:
        mean = np.mean(x)
        std = np.std(x)

        if std == 0:
            std = 1.0

        x_processed = (x - mean) / std

        scaler["mean"] = mean
        scaler["std"] = std
    else:
        x_processed = x.copy()

    # Add dummy feature x0 = 1
    X = np.column_stack((np.ones(x_processed.shape[0]), x_processed))

    return X, y, scaler


def compute_cost(X, y, theta):
    """
    Compute Mean Squared Error cost:

        J(theta) = (1 / 2m) * sum((X.theta - y)^2)
    """
    m = len(y)
    predictions = X @ theta
    errors = predictions - y

    cost = (1 / (2 * m)) * np.sum(errors ** 2)

    return cost


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=2000):
    """
    Optimize theta using batch gradient descent.

    Returns
    -------
    theta : np.ndarray
        Learned parameters.
    cost_history : np.ndarray
        Cost recorded at every iteration.
    """
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = X @ theta
        errors = predictions - y

        # Vectorized gradient calculation
        gradient = (1 / m) * (X.T @ errors)

        theta = theta - learning_rate * gradient
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history


def train(X, y, learning_rate=0.01, iterations=2000):
    """
    Initialize model parameters and train the model.
    """
    theta = np.zeros(X.shape[1])

    theta, cost_history = gradient_descent(
        X,
        y,
        theta,
        learning_rate=learning_rate,
        iterations=iterations
    )

    return theta, cost_history


def evaluate(X, y, theta):
    """
    Evaluate the trained model.

    Returns
    -------
    mse : float
        Mean Squared Error.
    rmse : float
        Root Mean Squared Error.
    r2 : float
        R-squared score.
    """
    predictions = X @ theta

    mse = np.mean((y - predictions) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return mse, rmse, r2


def convert_theta_to_original_scale(theta, scaler):
    """
    Convert learned parameters back to the original x scale.

    If z = (x - mean) / std and
        y = theta0 + theta1*z,

    then:
        y = intercept + slope*x
    """
    if not scaler["scaled"]:
        return theta[0], theta[1]

    mean = scaler["mean"]
    std = scaler["std"]

    slope = theta[1] / std
    intercept = theta[0] - (theta[1] * mean / std)

    return intercept, slope


def plot_data(x, y):
    """Plot the original dataset."""
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="Data points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Synthetic Dataset")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_error(cost_history):
    """Plot cost against gradient descent iterations."""
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(cost_history) + 1), cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(theta)")
    plt.title("Training Error Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_regression_line(x, y, intercept, slope):
    """Plot the dataset together with the learned regression line."""
    predictions = intercept + slope * x

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="Data points")
    plt.plot(x, predictions, label="Learned regression line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------
    x, y = load_data("dataset.csv")

    print(f"Number of samples: {len(x)}")

    # Plot raw data
    plot_data(x, y)

    # ---------------------------------------------------------
    # 2. Process data
    # ---------------------------------------------------------
    # Feature scaling is useful here because x ranges from 1 to 100.
    X, y, scaler = process_data(x, y, scale=True)

    # ---------------------------------------------------------
    # 3. Train model
    # ---------------------------------------------------------
    learning_rate = 0.05
    iterations = 2000

    theta, cost_history = train(
        X,
        y,
        learning_rate=learning_rate,
        iterations=iterations
    )

    # Convert parameters to original x scale for easier interpretation
    intercept, slope = convert_theta_to_original_scale(theta, scaler)

    print("\nLearned parameters:")
    print(f"theta (processed scale) = {theta}")
    print(f"Intercept = {intercept:.6f}")
    print(f"Slope     = {slope:.6f}")
    print(f"Learned equation: y = {intercept:.6f} + {slope:.6f}x")

    # ---------------------------------------------------------
    # 4. Evaluate
    # ---------------------------------------------------------
    mse, rmse, r2 = evaluate(X, y, theta)

    print("\nEvaluation:")
    print(f"MSE  = {mse:.6f}")
    print(f"RMSE = {rmse:.6f}")
    print(f"R^2  = {r2:.6f}")

    # ---------------------------------------------------------
    # 5. Plot training error and learned regression line
    # ---------------------------------------------------------
    plot_training_error(cost_history)
    plot_regression_line(x, y, intercept, slope)


if __name__ == "__main__":
    main()
