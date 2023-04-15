from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as mat
import sys
from tqdm import tqdm

def add_polynomial_features(x, power):

    if (not isinstance(x, np.ndarray) or not type(power) is int):
        return None
    if (power <=0):
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

class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        if (type(thetas) is list):
            try:
                thetas = np.array(thetas)
            except:
                return None

        if not (MyLinearRegression.check_matix(thetas)):
            return None
        if thetas.shape[1] != 1:
            return None
        if (not type(alpha) is float):
            return None
        if not (type(max_iter) is int):
            return None
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(thetas)

    @staticmethod
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

    @staticmethod
    def grad_(x, y, theta):
        X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)
        return(np.matmul(X_prime.T, (np.matmul(X_prime, theta) - y)) / x.shape[0])

    def fit_(self, x, y):
        if not MyLinearRegression.check_matix(x) or not MyLinearRegression.check_matix(y):
            return None
        if (y.shape[1] != 1):
            return None
        if (x.shape[0] != y.shape[0]):
            return None
        if (x.shape[1] != self.thetas.shape[0] - 1):
            return None

        for it in tqdm(range(self.max_iter)):
            grad = MyLinearRegression.grad_(x, y, self.thetas)
            if True in np.isnan(grad):
                return None
            self.thetas = self.thetas - (self.alpha * grad)
            if True in np.isnan(self.thetas):
                return None
        return self.thetas

    def predict_(self, x):
        if not MyLinearRegression.check_matix(x):
            return None

        if (x.shape[1] != self.thetas.shape[0] - 1):
            return None

        X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)
        return (np.matmul(X_prime, self.thetas).astype(float))

    def loss_elem_(self, y, y_hat):
        if not MyLinearRegression.check_matix(y) or not MyLinearRegression.check_matix(y_hat):
            return None
        if (y.shape[0] != y_hat.shape[0]):
            return None
        if (y.shape[1] != 1 or y_hat.shape[1] != 1):
            return None
        return((y_hat - y) * (y_hat - y))

    def loss_(self, y, y_hat):
        if not MyLinearRegression.check_matix(y) or not MyLinearRegression.check_matix(y_hat):
            return None
        if (y.shape[0] != y_hat.shape[0]):
            return None
        if (y.shape[1] != 1 or y_hat.shape[1] != 1):
            return None
        return((sum((y_hat - y) * (y_hat - y)) / (2 * y.shape[0]))[0])
    def mse_(y, y_hat):
        if not MyLinearRegression.check_matix(y) or not MyLinearRegression.check_matix(y_hat):
            return None
        if (y.shape[0] != y_hat.shape[0]):
            return None
        if (y.shape[1] != 1 or y_hat.shape[1] != 1):
            return None
        return((sum((y_hat - y) * (y_hat - y)) / (y.shape[0]))[0])

def load_data(path):
    if not type(path) is str:
        return None
    try:
        ret = pd.read_csv(path)
    except:
        return None
    print(f"Loading dataset from {path} of dimensions {ret.shape[0]} x {ret.shape[1]}", end='\n\n')
    return ret

def printhetas(thetas):
    ord = thetas.shape[0]
    res = ""
    for idx in reversed(range(1, ord)):
        res = res + f"{thetas[idx][0]} * x{idx} + "

    return (res + f"{thetas[0][0]}")

def getmodel(ord, thetas, min, max):
    x = np.linspace(min, max, 1000)
    y = np.zeros(x.shape[0])

    for it in range(1, ord + 1):
        y = y + thetas[it][0] * (x ** it)
    return (y + thetas[0][0])

def plot_order(X, Y, order, info):
    if (type(order) is not int):
        return None
    if (order < 1):
        return None

    mylr = MyLinearRegression(thetas=info["thetas"], alpha = info["alpha"], max_iter=info["iter"])
    if (mylr is None):
        return None
    X = add_polynomial_features(X, order)
    print(f"Polynomial Regression of Score, from Micrograms of order {order} fitting...")
    mylr.fit_(X, Y)
    print("resulting model : ", printhetas(mylr.thetas))
    mse = MyLinearRegression.mse_(Y, mylr.predict_(X))
    print(f"Mean Square Error of polynomial order {order} :", mse)

    plt.title("Polynomial Linear Regression, order %d\nMSE : %.2f" % (order, mse))
    plt.scatter(X[:, 0], Y, label="Score data")
    xplot = np.linspace(0, 7, 1000)
    yplot = getmodel(order, mylr.thetas, 0, 7)
    plt.plot(xplot, yplot, color='orange', label="Prediction model")
    plt.grid()
    plt.xlabel("Micrograms")
    plt.ylabel("Score")
    plt.legend(loc="upper right")
    plt.subplots_adjust(bottom=0.125)
    plt.show()
    return (mylr.thetas, mse)

def plot_all_order(X, Y, thetas_list):
    plt.title("Polynomial models comparaison")
    plt.scatter(X, Y, label ="Score data")
    xplot = np.linspace(1, 6.5, 1000)
    for order in range(1, 7):
        yplot = getmodel(order, thetas_list[order - 1], 1, 6.5)
        plt.plot(xplot, yplot, label=f"Model order {order}")
    plt.grid()
    plt.xlabel("Micrograms")
    plt.ylabel("Score")
    plt.legend(loc="upper right")
    plt.show()

def plot_mseorder(mse):
    plt.bar([1,2,3,4,5,6], mse, color='g')
    plt.title("Mean Square Error by the order of the Polynomial hypothesis")
    plt.xlabel("Polynomial order")
    plt.ylabel("MSE")
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    path_data = "are_blue_pills_magics.csv"
    df = load_data(path_data)
    if df is None:
        print(f"Error loading data from {path_data}")
        sys.exit()

    X = df[['Micrograms']].to_numpy()
    Y = df[['Score']].to_numpy()

    models = []

    models.append(plot_order(X, Y, 1, {"thetas" : [[80.], [-1]], "alpha" : 1e-4, "iter" : 100000}))
    models.append(plot_order(X, Y, 2, {"thetas" : [[90.], [-2.], [0.]], "alpha" : 1e-4, "iter" : 1000000}))
    models.append(plot_order(X, Y, 3, {"thetas" : [[80.], [-2.], [-1], [0.]], "alpha" : 5e-5, "iter" : 2000000}))
    models.append(plot_order(X, Y, 4, {"thetas" : [[-20.], [160.], [-80.], [10.], [-1.]], "alpha" : 1e-6, "iter" : 100000}))
    models.append(plot_order(X, Y, 5, {"thetas" : [[1140.], [-1850.], [1110.], [-305.], [40.], [-2.]], "alpha" : 1e-8, "iter" : 100000}))
    models.append(plot_order(X, Y, 6, {"thetas" : [[9110.], [-18015.], [13400.], [-4935.], [966.], [-96.4], [3.86]], "alpha" : 1e-9, "iter" : 100000}))

    plot_mseorder([x[1] for x in models])
    print("Greater is the order or the polynomial regression, lower is the mean square error, meaning better is the model.")
    plot_all_order(X, Y, [x[0] for x in models])
    print("However as we can see the high degree models must contort themself in order to fit the dataset")
    print("This means that these models only perform good on the training dataset, this is called overfitting, hence the need to test our models on a different dataset.")

















