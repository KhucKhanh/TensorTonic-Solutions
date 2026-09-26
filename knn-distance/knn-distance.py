import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    answer = []
    temp = [0] * k
    if len(X_test) == 0:
        return np.empty((0, k))
    for i in range(len(X_test)):
        check = [0] * len(X_train)
        for j in range(len(X_train)):
            check[j] = np.linalg.norm(X_test[i] - X_train[j])
        a = sorted(range(len(X_train)), key=lambda i: check[i])[:k]
        if len(a) < k:
            for _ in range(len(a), k):
                a.append(-1)
        answer.append(a)
    
    return np.array(answer)