import numpy as np
import math as mat

def check_matix(mat):
    if not (isinstance(mat, np.ndarray)):
        return False
    if mat.dtype != "int64" and mat.dtype != "float64":
        return False
    if len(mat.shape) != 2:
        return False
    if (mat.size == 0):
        return False
    return True

def sigmoid_(x):
    if (not isinstance(x, np.ndarray)):
        return None
    if (len(x.shape) != 2):
        return None
    if (x.shape[1] != 1):
        return None
    if (x.size == 0):
        return None

    sig = lambda x : 1/ (1 + mat.exp(-x))

    return (np.array([[sig(elem)] for elem in x]))

def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    if (not check_matix(x) or not check_matix(theta)):
        return None
    if theta.shape[1] != 1 or x.shape[1] != theta.shape[0] - 1:
        return None
    X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)
    return (sigmoid_(np.matmul(X_prime, theta)))
    

if __name__ == "__main__":
    print("Exemple 1")
    x = np.array([4]).reshape((-1, 1))
    theta = np.array([[2], [0.5]])
    print(logistic_predict_(x, theta), end="\n\n")

    print("Exemple 2")
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(logistic_predict_(x2, theta2), end="\n\n")

    print("Exemple 3")
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(logistic_predict_(x3, theta3), end="\n\n")