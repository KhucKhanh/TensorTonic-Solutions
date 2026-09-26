import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(X.shape[0], 1)

    w = np.zeros(X.shape[1])
    b = 0
    z = _sigmoid( X @ w + b)

    for i in range(steps):
        w = w + lr * X.T @ (y - z) / X.shape[0]
        b = b + lr * np.mean(y - z)
        z = _sigmoid( X @ w + b)

    return (w, b)