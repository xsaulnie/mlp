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

def reg_logistic_grad(y, x, theta, lambda_):
    """
    Computes the regularized logistic gradient of three non-empty numpy.ndarray, with two for-loops. The three array
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """

    if not check_matrix(y) or not check_matrix(x) or not check_matrix(theta):
        return None
    if y.shape[1] != 1 or x.shape[0] != y.shape[0] or theta.shape[1] != 1:
        return None
    if x.shape[1] != theta.shape[0] - 1:
        return None 
    if (not type(lambda_) is float and not type(lambda_) is int):
        return None

    m = x.shape[0]
    n = x.shape[1]

    def pred(x, theta):
        ret = 0
        sig = lambda x : 1 / (1 + mat.exp(-x))
        for idx in range(len(x)):
            ret = ret + x[idx] * theta[idx + 1][0]
        return (sig(ret + theta[0][0]))

    grad = []
    grad0 = 0
    for i in range (m):
        grad0 = grad0 + pred(x[i], theta) - y[i][0]
    grad.append(grad0 / m)

    for j in range (n):
        gradx = 0
        for i in range(m):
            gradx = gradx + (pred(x[i], theta) - y[i][0]) * x[i][j] 
        grad.append((gradx + lambda_ * theta[j + 1][0] ) / m)

    return(np.atleast_2d(np.array(grad)).T)

def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any for-loop. The three arr
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of shape m * n.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """
    if not check_matrix(y) or not check_matrix(x) or not check_matrix(theta):
        return None
    if y.shape[1] != 1 or x.shape[0] != y.shape[0] or theta.shape[1] != 1:
        return None
    if x.shape[1] != theta.shape[0] - 1:
        return None 
    if (not type(lambda_) is float and not type(lambda_) is int):
        return None

    m = x.shape[0]
    X_prime = np.concatenate((np.ones((m, 1)), x), axis=1).astype(float)
    theta_prime = np.copy(theta).astype(float)
    theta_prime[0][0] = 0
    def sigmoid(x):
        sig = lambda x : 1 / (1 + mat.exp(-x))
        return (np.array([[sig(elem)] for elem in x]))

    return ((np.matmul(X_prime.T, (sigmoid(np.matmul(X_prime, theta)) - y)) + lambda_ * theta_prime) / m)    

if __name__ == "__main__":
    x = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    print("Exemple from subject\n")

    print("x matrix\n")
    print(x, end="\n\n")
    print("y matrix\n")
    print(y, end="\n\n")
    print("theta matrix\n")
    print(theta, end="\n\n")

    print("Exemple 1.1\n")
    print(reg_logistic_grad(y, x, theta, 1), end="\n\n")
    print("Exemple 1.2\n")
    print(vec_reg_logistic_grad(y, x, theta, 1), end="\n\n")
    print("Exemple 2.1\n")
    print(reg_logistic_grad(y, x, theta, 0.5), end="\n\n")
    print("Exemple 2.2\n")
    print(vec_reg_logistic_grad(y, x, theta, 0.5), end="\n\n")
    print("Exemple 3.1\n")
    print(reg_logistic_grad(y, x, theta, 0.0), end="\n\n")
    print("Exemple 3.2\n")
    print(vec_reg_logistic_grad(y, x, theta, 0.0), end="\n\n")
