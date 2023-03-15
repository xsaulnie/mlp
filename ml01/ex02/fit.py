import numpy as np
import math as mat

def vec_col(vec):
    if not (isinstance(vec, np.ndarray)):
        return False
    if (vec.dtype != "float64" and vec.dtype != "int64"):
        return False
    if len(vec.shape) != 2:
        return False
    if vec.size == 0:
        return False
    if (vec.shape[1] != 1):
        return False
    return True

def predict(x, theta):
    if not vec_col(x) or not vec_col(theta):
        return None
    if theta.shape[0] != 2:
        return None
    X_prime = np.concatenate((np.ones(x.shape[0], dtype=x.dtype).reshape(x.shape[0], 1), x), axis=1)
    return (np.matmul(X_prime, theta).astype(float))
    

def simple_gradient(x, y, theta):
    if not vec_col(x) or not vec_col(y) or not vec_col(theta):
        return None
    if theta.shape[0] != 2 or x.shape[0] != y.shape[0]:
        return None

    X_prime = np.concatenate((np.ones(x.shape[0], dtype=x.dtype).reshape(x.shape[0], 1), x), axis=1)

    grad = np.matmul(X_prime.T, np.matmul(X_prime, theta).astype(float) - y) / y.shape[0]

    return grad

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """

    if (not type(max_iter) is int):
        return None
    if (not type(alpha) is float):
        return None
    if not vec_col(x) or not vec_col(y) or not vec_col(theta):
        return None
    if theta.shape[0] != 2 or x.shape[0] != y.shape[0]:
        return None
    new_theta = np.array([theta[0], theta[1]]).astype(float)

    for it in range(max_iter):
        grad = simple_gradient(x, y, new_theta)
        new_theta[0][0] = new_theta[0][0] - (alpha * grad[0][0])
        if mat.isnan(new_theta[0][0]):
            return None
        new_theta[1][0] = new_theta[1][0] - (alpha * grad[1][0])
        if mat.isnan(new_theta[1][0]):
            return None
    return new_theta

if __name__ == "__main__":
    print("Main exemple")
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    theta= np.array([1, 1]).reshape((-1, 1))

    print("Value of x and y : ")
    print(x)
    print(y)
    print("fiting...")
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print("theta after fit : ")
    print(theta1)

    print("linar regression of y : ")

    print(predict(x, theta1))
    print("Other exemple : ")
    print("10 random values for x and y = 3.76 * x + 2.28")
    print("fiting...")
    randx = np.random.rand(10, 1)
    ylin = randx * 3.76 + 2.28

    res = fit_(randx, ylin, np.array([[1.], [1.]]), alpha=5e-3, max_iter=1500000)
    print("theta founded : ", res)
    print(f"relation y= {res[1][0]} * x + {res[0][0]}")

