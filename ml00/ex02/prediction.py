import numpy as np
import sys
def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """


    if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)):
        return None
    if not(x.dtype == "int64" or x.dtype == "float64"):
        return None

    if (len(x.shape) != 1):
        return None

    if (len(theta) != 2):
        return None

    y_hat = [0] * len(x)

    for idx in range(len(x)):
        y_hat[idx] = theta[0] + theta[1] * x[idx]
    return np.array(y_hat, dtype=float)

if __name__ == "__main__":

    x = np.arange(1, 6)

    theta1 = np.array([5, 0])
    print(simple_predict(x, theta1))

    theta2 = np.array([0, 1])
    print(simple_predict(x, theta2))

    theta3 = np.array([5, 3])
    print(simple_predict(x, theta3))

    theta4 = np.array([-3, 1])
    print(simple_predict(x, theta4))

    
