import numpy as np

def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array of dimension m * n.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    if not (isinstance(x, np.ndarray)):
        return None
    if (x.size == 0):
        return None
    if (len(x.shape) == 1):
        x = np.atleast_2d(x).T

    a = np.ones(x.shape[0], dtype=x.dtype).reshape(x.shape[0], 1)
    res = np.concatenate((a, x), axis=1)
    return (res.astype(float))

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exceptions.
    """

    if not (isinstance(x, np.ndarray)) or not (isinstance(theta, np.ndarray)):
        return None
    if (len(x.shape) > 2 or len(theta.shape) > 2):
        return None
    
    if (x.size == 0 or theta.size == 0):
        return None
    if (len(theta.shape) == 1):
        theta = np.atleast_2d(theta).T
    if (len(x.shape) == 1):
        x = np.atleast_2d(x).T
    if (x.shape[1] != 1 or theta.shape[0] != 2 or theta.shape[1] != 1):
        return None

    return (np.matmul(add_intercept(x), theta).astype(float))

    
if __name__ == "__main__":
    x = np.arange(1, 6)

    print("Exemple1")
    theta1 = np.array([[5], [0]])
    print(predict_(x, theta1))
    print()
    print("Exemple2")
    theta1 = np.array([[0], [1]])
    print(predict_(x, theta1))
    print()
    print("Exemple3")
    theta1 = np.array([[5], [3]])
    print(predict_(x, theta1))
    print()
    print("Exemple4")
    theta1 = np.array([[-3], [1]])
    print(predict_(x, theta1))
