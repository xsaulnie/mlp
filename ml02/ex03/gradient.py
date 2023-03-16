import numpy as np

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

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
        The gradient as a numpy.array, a vector of dimensions n * 1, #(+1 ?)
        containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible dimensions.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not check_matix(x) or not check_matix(y) or not check_matix(theta):
        return None
    if x.shape[0] != y.shape[0]:
        return None
    if theta.shape[1] != 1 or y.shape[1] != 1:
        return None
    if x.shape[1] != theta.shape[0] - 1:
        return None

    X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)
    grad = np.matmul(X_prime.T, (np.matmul(X_prime, theta) - y)) / x.shape[0]
    return (grad)

if __name__ == "__main__":
    x = np.array([
    [ -6, -7, -9],
    [ 13, -2, 14],
    [ -7, 14, -1],
    [ -8, -4, 6],
    [ -5, -9, 6],
    [ 1, -5, 11],
    [ 9, -11, 8]])

    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print("Exemple 1")
    theta1 = np.array([0,3,0.5,-6]).reshape((-1, 1))
    print(gradient(x, y, theta1))

    print("Exemple 2")
    theta2 = np.array([0,0,0,0]).reshape((-1, 1))
    print(gradient(x, y, theta2))