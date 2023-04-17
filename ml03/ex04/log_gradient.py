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
    sig = lambda x : 1/ (1 + mat.exp(-x))
    return (np.array([[sig(elem)] for elem in x]))

def logistic_predict_(x, theta):
    if (not check_matix(x) or not check_matix(theta)):
        return None
    if theta.shape[1] != 1 or x.shape[1] != theta.shape[0] - 1:
        return None
    res = []
    for i in range(x.shape[0]):
        ret = 0
        for j in range(x.shape[1]):
            ret = ret + x[i][j] * theta[j + 1][0]
        res.append(ret + theta[0][0])
    return (sigmoid_(np.array(res)))


def log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatiblArgs:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """

    if not(isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray)):
        return None
    if (y.shape[1] != 1 or x.shape[1] != theta.shape[0] - 1 or theta.shape[1] != 1):
        return None
    grad = []
    y_hat = logistic_predict_(x, theta)

    for i in range(x.shape[1] + 1):
        ret = 0
        for j in range(x.shape[0]):
            if (i != 0):
                ret = ret + (y_hat[j] - y[j]) * x[j][i - 1]
            else:
                ret = ret + (y_hat[j] - y[j])
        grad.append(ret / x.shape[0])
    return  np.array(grad)

if __name__ == "__main__":
    print("Exemple 1")
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    print(log_gradient(x1, y1, theta1), end="\n\n")

    print("Exemple 2")
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(log_gradient(x2, y2, theta2), end="\n\n")

    print("Exemple 3")
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(log_gradient(x3, y3, theta3), end="\n\n")