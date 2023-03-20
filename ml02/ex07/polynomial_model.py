import numpy as np
import math as mat
def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
        The matrix of polynomial features as a numpy.array, of dimension m * n,
        containing the polynomial feature values for all training examples.
        None if x is an empty numpy.array.
        None if x or power is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if (not isinstance(x, np.ndarray) or not type(power) is int):
        return None
    if (x.dtype != "int64" and x.dtype != "float64"):
        return None
    if (len(x.shape) != 2):
        return None
    if (x.size == 0):
        return None
    if x.shape[1] != 1:
        return None
    ret = np.zeros((x.shape[0], power), dtype=x.dtype)
    for lin in range(x.shape[0]):
        for col in range(power):
            ret[lin][col] = mat.pow(x[lin], col + 1)
    return ret

    
if __name__ == "__main__":
    x = np.arange(1, 6).reshape(-1, 1)
    print("Exemple 1")
    print(add_polynomial_features(x, 3))
    print('Exemple 2')
    print(add_polynomial_features(x, 6))