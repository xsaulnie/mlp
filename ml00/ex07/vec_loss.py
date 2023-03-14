from matplotlib import pyplot as plt
import numpy as np

def is_vector_column(vec):
    if not (isinstance(vec, np.ndarray)):
        return False
    if (vec.dtype != "float64" and vec.dtype != "int64"):
        return False
    if len(vec.shape) != 2:
        return False
    if vec.size == 0:
        return False
    if (vec.shape[1] != 1):
        return False
    return True


def loss_(y, y_hat):
    """Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    The half mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    Raises:
    This function should not raise any Exceptions.
    """

    if (not is_vector_column(y) or not is_vector_column(y_hat)):
        return None
    if (y.shape[0] != y_hat.shape[0]):
        return None
    
    HMS = np.sum(((y_hat - y) * (y_hat - y)) / (2 * y.shape[0]))
    return HMS

if __name__ == "__main__":
    print("Main exemple")
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

    print(loss_(X, Y))

    print(loss_(X, X))


