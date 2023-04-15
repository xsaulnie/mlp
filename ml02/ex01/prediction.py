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

def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimensions m * n.
        theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
    Return:
        y_hat as a numpy.array, a vector of dimensions m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
        None if x or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not check_matix(x) or not check_matix(theta):
        return None
    if (x.shape[1] != theta.shape[0] - 1):
        return None
    if (theta.shape[1] != 1):
        return None

    X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)

    return np.matmul(X_prime, theta).astype(float)


if __name__ == "__main__":
    x = np.arange(1,13).reshape((4,-1))
    theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))

    print("X matrix")
    print(x)

    print("Exemple 1")
    print(predict_(x, theta1))

    theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
    print("Exemple 2")
    print(predict_(x, theta2))

    theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
    print("Exemple 3")
    print(predict_(x, theta3))

    theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
    print("Exemple 4")
    print(predict_(x, theta4))