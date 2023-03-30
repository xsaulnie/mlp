import numpy as np
import math as mat

def check_matrix(mat):
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

    sig = lambda x : 1 / (1 + mat.exp(-x))

    return (np.array([[sig(elem)] for elem in x]))

def logistic_predict_(x, theta):
    if (not check_matrix(x) or not check_matrix(theta)):
        return None
    if theta.shape[1] != 1 or x.shape[1] != theta.shape[0] - 1:
        return None
    X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)
    return (sigmoid_(np.matmul(X_prime, theta)))

def vec_log_loss_(y, y_hat, eps=1e-15):
    """
    Compute the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """

    if (not check_matrix(y) or not check_matrix(y_hat) or not type(eps) is float):
        return None
    if not y.shape[1] == 1 or not y_hat.shape[1] == 1:
        return None
    if not y.shape[0] == y_hat.shape[0]:
        return None

    logv = np.vectorize(mat.log)
    return ( -1/ y.shape[0] * ( np.dot(y.T, logv(y_hat)) + np.dot ((np.ones((y.shape[0], 1)) - y).T, logv( np.ones((y.shape[0], 1)) - y_hat))))[0][0]

if __name__ == "__main__":
    print("Exemple 1")
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    y_hat1 = logistic_predict_(x1, theta1)
    print(vec_log_loss_(y1, y_hat1))

    print("Exemple 2")
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat2 = logistic_predict_(x2, theta2)
    print(vec_log_loss_(y2, y_hat2))

    print("Exemple 3")
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    y_hat3 = logistic_predict_(x3, theta3)
    print(vec_log_loss_(y3, y_hat3))

