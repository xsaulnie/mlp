import math as mat
import numpy as np
def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """

    if (not isinstance(x, np.ndarray)):
        return None
    if x.dtype != "int64" and x.dtype != "float64":
        return None
    if (len(x.shape) != 2):
        return None
    if (x.shape[1] != 1):
        return None
    if (x.size == 0):
        return None

    sig = lambda x : 1/ (1 + mat.exp(-x))

    return (np.array([[sig(elem)] for elem in x]))

if __name__ == "__main__":
    print("Exemple 1")
    x = np.array([[-4]])
    print(sigmoid_(x))
    print("Exemple 2")
    x = np.array([[2]])
    print(sigmoid_(x))
    print("Exemple 3")
    x = np.array([[-4], [2], [0]])
    print(sigmoid_(x))
