import numpy as np

def check_matrix(mat):
    if not isinstance(mat, np.ndarray):
        return False
    if mat.dtype != "int64" and mat.dtype != "float64":
        return False
    if len(mat.shape) != 2:
        return False
    if mat.size == 0:
        return False
    return True

def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not check_matrix(theta):
        return None
    if theta.shape[1] != 1:
        return None

    ret = np.zeros((theta.shape[0] - 1, 1))

    for lin in range(theta.shape[0] - 1):
        ret[lin] = theta[lin + 1] * theta[lin + 1]
    return np.sum(ret)

def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not check_matrix(theta):
        return None
    if theta.shape[1] != 1:
        return None
    theta_prime = np.copy(theta).astype(float)
    theta_prime[0][0] = 0
    return (np.sum(np.dot(theta_prime.T, theta_prime)))

if __name__ == "__main__":
    print("Exemple from subject\n")
    x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print("x matrix\n")
    print(x, end="\n\n")
    print("Exemple 1\n")
    print(iterative_l2(x), end="\n\n")
    print("Exemple 2\n")
    print(l2(x), end="\n\n")
    y = np.array([3,0.5,-6]).reshape((-1, 1))
    print("y matrix\n")
    print(y, end="\n\n")
    print("Exemple 3\n")
    print(iterative_l2(y), end="\n\n")
    print("Exemple 4\n")
    print(l2(y))

