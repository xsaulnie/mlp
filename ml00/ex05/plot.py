from matplotlib import pyplot as plt 
import numpy as np

def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    y: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exceptions.
    """

    if not (isinstance(x, np.ndarray)) or not (isinstance(theta, np.ndarray) or not (isinstance(y, np.ndarray))):
        return None
    if (len(x.shape) > 2 or len(y.shape) > 2 or len(theta.shape) > 2):
        return None
    if (x.size == 0 or theta.size == 0 or y.size == 0):
        return None
    if (len(theta.shape) == 1):
        theta = np.atleast_2d(theta).T
    if (len(x.shape) == 1):
        x = np.atleast_2d(x).T
    if (len(y.shape) == 1):
        y = np.atleast_2d(y).T
    if (x.shape[1] != 1 or theta.shape[0] != 2 or theta.shape[1] != 1 or y.shape[1] != 1):
        return None
    plt.scatter(x, y)

    print(theta[0], theta[1])

    xplot = np.linspace(np.amin(x), np.amax(x), 100)

    print(xplot)
    yplot = xplot * theta[1][0] + theta[0][0]
    print(yplot)
    plt.plot(xplot, yplot, color='orange')
    plt.show()


if __name__ == "__main__":
    x = np.arange(1, 6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])

    theta1 = np.array([[4.5], [-0.2]])
    plot(x, y, theta1)

    theta2 = np.array([[-1.5], [2]])
    plot(x, y, theta2)

    theta3 = np.array([[3], [0.3]])
    plot(x, y, theta3)
    
