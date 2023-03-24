import numpy as np
import math as mat

def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x' as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
        This function shouldn't raise any Exception.
    """

    if not (isinstance(x, np.ndarray)):
        return None
    if (x.dtype != "float64" and x.dtype != "int64"):
        return None
    if (len(x.shape) > 2):
        return None
    if (len(x.shape) == 1):
        x = np.atleast_2d(x).T
    if x.size == 0 or x.size == 1:
        return None
    if x.shape[1] != 1:
        return None

    mean = sum(x)[0] / x.shape[0]

    std = mat.sqrt(sum((x - np.full((x.shape[0], 1), mean).astype(float))**2) / (x.shape[0])) # -1 ?

    normi = lambda xi : (xi - mean) / std

    return (normi(x).T[0])

if __name__ == "__main__":


    print("Exemple 1")
    X = np.array([0, 15, -9, 7, 12, 3, -21])

    print("zscore X")
    print(zscore(X).__repr__(), end="\n\n")

    print("Exemple 2")
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

    print("zcore Y")
    print(zscore(Y).__repr__())



