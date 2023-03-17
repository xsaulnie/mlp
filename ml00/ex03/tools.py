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

if __name__ == "__main__":
    x = np.arange(1, 6)
    y = np.arange(1,10).reshape((3,3))

    print(add_intercept(x))
    print(add_intercept(y))
