from matplotlib import pyplot as plt 
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

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exceptions.
    """

    if not (isinstance(x, np.ndarray)) or not (isinstance(theta, np.ndarray)):
        return None
    if (len(x.shape) > 2 or len(theta.shape) > 2):
        return None
    
    if (x.size == 0 or theta.size == 0):
        return None
    if (len(theta.shape) == 1):
        theta = np.atleast_2d(theta).T
    if (len(x.shape) == 1):
        x = np.atleast_2d(x).T
    if (x.shape[1] != 1 or theta.shape[0] != 2 or theta.shape[1] != 1):
        return None

    return (np.matmul(add_intercept(x), theta).astype(float))

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


def loss_elem_(y, y_hat):
    """
    Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_elem: numpy.array, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
        Raises:
        This function should not raise any Exception.
    """
    if (not is_vector_column(y) or not is_vector_column(y_hat)):
        return None
    if (y.shape[0] != y_hat.shape[0]):
        return None
    J_elem = np.zeros((y.shape[0], 1))

    for idx in range(y.shape[0]):
        J_elem[idx][0] = (y_hat[idx][0] - y[idx][0]) ** 2 
    return J_elem.astype(float)

def loss_(y, y_hat):
    """
    Description:
        Calculates the value of loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """

    if (not is_vector_column(y) or not is_vector_column(y_hat)):
        return None
    if (y.shape[0] != y_hat.shape[0]):
        return None
    J_elem = loss_elem_(y, y_hat)
    return np.sum(J_elem) / (2 * J_elem.shape[0])


if __name__ == "__main__":
    x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])
    y_hat1 = predict_(x1, theta1)
    y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

    x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
    theta2 = np.array([[0.], [1.]]).reshape(-1, 1)
    y_hat2 = predict_(x2, theta2)
    y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)

    print("Main Exemple")
    print("Exemple 1 loss_elem(y1, y_hat1)")
    print(loss_elem_(y1, y_hat1))
    print("Exemple 2 loss_(y1, y_hat1)")
    print(loss_(y1, y_hat1))

    print("Exemple 3 loss_(y2, y_hat2)")
    print(loss_(y2, y_hat2))

    print("Exemple 4 (loss y2, y2) : same data, no loss")
    print(loss_(y2, y2))
