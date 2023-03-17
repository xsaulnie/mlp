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

def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    """

    if not (isinstance(x, np.ndarray)) or not (isinstance(theta, np.ndarray) or not isinstance(y, np.ndarray)):
        return None
    if (len(x.shape) > 2 or len(theta.shape) > 2 or len(y.shape) > 2):
        return None
    if (x.dtype != "float64" and x.dtype != "int64"):
        return None 
    if (y.dtype != "float64" and y.dtype != "int64"):
        return None 
    if (theta.dtype != "float64" and theta.dtype != "int64"):
        return None 
    if (x.size == 0 or theta.size == 0 or y.size == 0):
        return None
    if (len(theta.shape) == 1):
        theta = np.atleast_2d(theta).T
    if (len(x.shape) == 1):
        x = np.atleast_2d(x).T
    if len(y.shape) == 1:
        y = np.atleast_2d(y).T
    if (x.shape[1] != 1 or theta.shape[0] != 2 or theta.shape[1] != 1 or y.shape[1] != 1):
        return None
    if (x.shape[0] != y.shape[0]):
        return None

    X_prime = np.concatenate((np.ones(x.shape[0], dtype=x.dtype).reshape(x.shape[0], 1), x), axis=1)
    y_hat = np.matmul(X_prime, theta).astype(float)
    cost = np.sum(((y_hat - y) * (y_hat - y)) / (y.shape[0]))

    fig = plt.figure()
    plt.title("Cost: %.6f" % (cost))
    plt.scatter(x, y)
    xplot = np.linspace(np.amin(x), np.amax(x), 100)
    yplot = xplot * theta[1][0] + theta[0][0]
    plt.plot(xplot, yplot, color ='orange')
    for idx in range(x.shape[0]):
        plt.vlines(x[idx], y[idx], y_hat[idx], color='red', linestyles='dashed')
    plt.show()


if __name__ == "__main__":

    print("Main Exemple")
    x = np.arange(1,6)
    y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])

    theta1=np.array([18, -1])
    plot_with_loss(x, y, theta1)

    theta2 = np.array([14, 0])
    plot_with_loss(x, y, theta2)

    theta3 = np.array([12, 0.8])
    plot_with_loss(x, y, theta3)

