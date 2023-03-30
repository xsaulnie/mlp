import numpy as np

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

def l2(theta):
    if not check_matrix(theta):
        return None
    if theta.shape[1] != 1:
        return None
    theta_prime = np.copy(theta).astype(float)
    theta_prime[0][0] = 0
    return (np.sum(np.dot(theta_prime.T, theta_prime)))

def reg_loss_(y, y_hat, theta, lambda_):
    """
    Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop.Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
    The regularized loss as a float.
        None if y, y_hat, or theta are empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """

    if not check_matrix(y) or not check_matrix(y_hat) or not check_matrix(theta):
        return None
    if (y.shape[0] != y_hat.shape[0]):
        return None
    if y.shape[1] != 1 or y_hat.shape[1] != 1 or theta.shape[1] != 1:
        return None
    if (not type(lambda_) is float):
        return None

    return ((np.dot((y_hat - y).T, (y_hat - y)) + lambda_ * l2(theta)) / (2 * y.shape[0]))[0][0]

if __name__ == "__main__":
    print("Exemple from subject\n")
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
    print("y matrix\n")
    print(y, end="\n\n")
    print("y_hat matrix\n")
    print(y_hat, end="\n\n")
    print("theta matrix\n")
    print(theta, end="\n\n")
    print("Exemple 1\n")
    print(reg_loss_(y, y_hat, theta, .5), end="\n\n")
    print("Exemple 2\n")
    print(reg_loss_(y, y_hat, theta, .05), end="\n\n")
    print("Exemple 3\n")
    print(reg_loss_(y, y_hat, theta, .9), end="\n\n")