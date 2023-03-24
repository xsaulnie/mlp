import numpy as np

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

def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for loop.
    The three arrays must have compatible shapes.
    Args:
        x: has to be a numpy.array, a matrix of shape m * 1.
        y: has to be a numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        None if x, y, or theta is an empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """

    if not vec_col(x) or not vec_col(y) or not vec_col(theta):
        return None
    if theta.shape[0] != 2 or x.shape[0] != y.shape[0]:
        return None

    X_prime = np.concatenate((np.ones(x.shape[0], dtype=x.dtype).reshape(x.shape[0], 1), x), axis=1)

    grad = np.matmul(X_prime.T, np.matmul(X_prime, theta).astype(float) - y) / y.shape[0]

    return grad


if __name__ == "__main__":
    print("Main exemple")
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))

    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    theta2 = np.array([1, -0.4]).reshape((-1, 1))

    print("first exemple")
    print(simple_gradient(x, y, theta1))
    print("second exemple")
    print(simple_gradient(x, y, theta2))
