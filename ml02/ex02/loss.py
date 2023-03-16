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

def loss_(y, y_hat):
    """Computes the mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Return:
        The mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
        None if y or y_hat is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not check_matix(y) or not check_matix(y_hat):
        return None
    if (y.shape[0] != y_hat.shape[0]):
        return None
    if (y.shape[1] != 1):
        return None
    if (y_hat.shape[1] != 1):
        return None

    return(sum((y_hat - y) * (y_hat - y))[0] / (2 * y.shape[0]))

if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

    print("Exemple 1")
    print(loss_(X, Y))
    print("Exemple 2")
    print(loss_(X, X))
    