import numpy as np
def minmax(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
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

    amin = np.amin(x)
    amax = np.amax(x)

    normi = lambda xi : (xi - amin) / (amax - amin)

    return(normi(x))



if __name__== "__main__":
    print("Main exemple")
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    print("minmax x")
    print(minmax(X))

    print("minmax Y")
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print(minmax(Y))

