from matplotlib import pyplot as plt
import numpy as np
import math as mat
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def is_vector_column(vec):
    if not (isinstance(vec, np.ndarray)):
        return False
    if (vec.dtype != "float64" and vec.dtype != "int64"):
        return False
    if vec.size == 0:
        return False
    if (len(vec.shape) == 1):
        return True
    if len(vec.shape) != 2:
        return False
    if (vec.shape[1] != 1):
        return False
    return True

def mse_(y, y_hat):
    """
    Description:
        Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not is_vector_column(y) or not is_vector_column(y_hat):
        return (None)

    if y.shape[0] != y_hat.shape[0]:
        return (None)

    mse = np.sum(((y_hat - y) * (y_hat - y)) / (y.shape[0]))

    return (mse)

def rmse_(y, y_hat):
    """
    Description:
        Calculate the RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """

    if not is_vector_column(y) or not is_vector_column(y_hat):
        return (None)

    if y.shape[0] != y_hat.shape[0]:
        return (None)

    rmse = mat.sqrt(np.sum(((y_hat - y) * (y_hat - y)) / (y.shape[0])))

    return (rmse)

def mae_(y, y_hat):
    """
    Description:
        Calculate the MAE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mae: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """

    if not is_vector_column(y) or not is_vector_column(y_hat):
        return (None)

    if y.shape[0] != y_hat.shape[0]:
        return (None)

    absvec = np.vectorize(abs)

    mae = np.sum((absvec(y_hat - y)) / (y.shape[0]))

    return (mae)

def r2score_(y, y_hat):
    """
    Description:
        Calculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not is_vector_column(y) or not is_vector_column(y_hat):
        return (None)

    if y.shape[0] != y_hat.shape[0]:
        return (None)

    y_mean = sum(y) / y.shape[0]

    r2score = 1 - (np.sum((y_hat - y) * (y_hat - y)) / np.sum((y - y_mean) * (y - y_mean)))

    return (r2score)

if __name__ == "__main__":
    print("Main exemple\n")

    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])

    print("mse")
    print(mse_(x, y))
    print("ref mse")
    print(mean_squared_error(x, y))

    print("rmse")
    print(rmse_(x, y))
    print("ref rmse")
    print(mat.sqrt(mean_squared_error(x, y)))

    print("mae")
    print(mae_(x, y))
    print("ref mae")
    print(mean_absolute_error(x, y))

    print("r2_score")
    print(r2score_(x, y))
    print("ref r2_score")
    print(r2_score(x, y))

