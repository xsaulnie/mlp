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
    

def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power give
    Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        power: has to be an int, the power up to which the columns of matrix x are going to be raised.
    Returns:
        The matrix of polynomial features as a numpy.ndarray, of shape m * (np), containg the polynomial feature vaNone if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """

    if not check_matrix(x) or not type(power) is int:
        return None
    ret = []
    for lin in range(x.shape[0]):
        pows = []
        for col in range(x.shape[1] * power):
            pows.append(x[lin][col % x.shape[1]] ** (int(col / x.shape[1]) + 1))
        ret.append(pows)

    return (np.array(ret))
    

if __name__ == "__main__":
    print("Exemple from subject\n")
    x = np.arange(1, 11).reshape(5, 2)
    print("x matrix\n")
    print(x)
    print()
    print("Exemple 1\n")
    print(add_polynomial_features(x, 3))
    print()
    print("Exemple 2\n")
    print(add_polynomial_features(x, 4))


    