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

def predict_(x, theta):
    if not check_matix(x) or not check_matix(theta):
        return None
    if (x.shape[1] != theta.shape[0] - 1):
        return None
    if (theta.shape[1] != 1):
        return None

    X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)

    return np.matmul(X_prime, theta).astype(float)

def gradient_(x, y, theta):
    X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)
    return(np.matmul(X_prime.T, (np.matmul(X_prime, theta) - y)) / x.shape[0])


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a matrix of dimension m * n:
        (number of training examples, number of features).
        y: has to be a numpy.array, a vector of dimension m * 1:
        (number of training examples, 1).
        theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
        (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
        new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
        This function should not raise any Exception.
    """

    if not check_matix(x) or not check_matix(y) or not check_matix(theta):
        return None
    if not type(alpha) is float or not type(max_iter) is int:
        return None
    if theta.shape[1] != 1 or y.shape[1] != 1:
        return None
    if (x.shape[0] != y.shape[0]):
        return None
    if (x.shape[1] != theta.shape[0] - 1):
        return None

    for it in range(max_iter):
        theta = theta - alpha * gradient_(x, y, theta)
        if True in np.isnan(theta):
            return None
    return (theta)

if __name__ == "__main__":

    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])
    print("Exemple 1")
    theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter=42000)
    print(theta2)

    print("Exemple 2")
    print(predict_(x, theta2))


    xrand = np.random.rand(10, 3)
    relation = np.array([[2.], [1.76], [-2.24], [0.86]])
    yrel = np.matmul(np.concatenate((np.ones((10,1)),xrand), axis=1).astype(float), relation)

    print("creating y = 2 + 1.76*x1 - 2.24*x2 + 0.86*x3 from a random x")

    thetafound = fit_(xrand, yrel, np.zeros((4, 1)), alpha=0.02, max_iter=100000)
    print("theta found after fit(alpha=0.2, iter=100000) from a theta null : ")
    print(thetafound)

