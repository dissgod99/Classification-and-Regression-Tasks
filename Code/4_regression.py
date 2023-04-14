#!/usr/bin/env python
# coding: utf-8
from matplotlib import pyplot as plt
import numpy as np


def visualize_data(X: np.ndarray, y: np.ndarray) -> None:
    """Visualizes the data points in a scatter plot."""
    plt.scatter(x=X, y=y)
    plt.title("Scatter plot of the provided data")
    plt.xlabel("X values")
    plt.ylabel("y values")
    plt.legend()
    plt.show()


def perform_linear_regression(Xp: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes the regression coefficients beta_hat."""
    return np.linalg.inv(Xp.T @ Xp) @ Xp.T @ y


def compute_predictions(Xp: np.ndarray, beta_hat: np.ndarray) -> np.ndarray:
    """Computes the corresponding prediction on the regression line for Xp."""
    raise NotImplementedError


def compute_mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the mean squared error."""
    raise NotImplementedError


def main():
    # Generate data
    n = 100
    X = np.random.rand(n)

    # Generate target with noise
    y = 3 * X + np.random.randn(n) * 0.3 + np.random.randn() * 5

    # Plot data
    visualize_data(X, y)

    # Add ones for intercept term
    Xp = np.c_[np.ones(n), X]
    beta_hat = perform_linear_regression(Xp, y)
    y_pred = compute_predictions(Xp, beta_hat)

    # Plot regression line over data
    plt.figure()
    plt.plot(X, y_pred, c="r", label="Regression line")
    visualize_data(X, y)

    # Compute mean squared error
    mse = compute_mse(y, y_pred)
    print(f"Mean Squared Error: {mse:2.4f}")


if __name__ == '__main__':
    main()
