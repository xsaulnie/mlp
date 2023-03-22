from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as mat
import sys
from tqdm import tqdm

def minmax(x):
    if not (isinstance(x, np.ndarray)):
        return None
    if (x.dtype != "float64" and x.dtype != "int64"):
        return None
    if (len(x.shape) == 1):
        x = np.atleast_2d(x).T

    amin = np.amin(x)
    amax = np.amax(x)

    normi = lambda xi : (xi - amin) / (amax - amin)

    return(normi(x))

def antiminmax(x):
    if not (isinstance(x, np.ndarray)):
        return None
    if (x.dtype != "float64" and x.dtype != "int64"):
        return None
    if (len(x.shape) == 1):
        x = np.atleast_2d(x).T

    amin = np.amin(x)
    amax = np.amax(x)

    antinormi = lambda xi : xi * (amax - amin) + amin

    return(antinormi(x))

def polynomial_features(x, power):

    if (not isinstance(x, np.ndarray) or not type(power) is int):
        return None
    if (x.dtype != "int64" and x.dtype != "float64"):
        return None
    if (len(x.shape) != 2):
        return None
    if (x.size == 0):
        return None

    if (power < 1):
        return None
    if (power == 1):
        return x
    ret = np.zeros((x.shape[0], power * x.shape[1]), dtype=x.dtype)
    for lin in range(x.shape[0]):
        for col in range(ret.shape[1]):
            ret[lin][col] = mat.pow(x[lin][col % x.shape[1]], col + 1)
    return ret

def data_spliter(x, y, proportion):

    data = np.concatenate((x, y), axis = 1)
    np.random.shuffle(data)

    nb = int(proportion * data.shape[0])

    return((data[0:nb,:-1], data[nb:,:-1], data[0:nb,-1:], data[nb:,-1:]))

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
    print(f"Loading dataset of dimensions {ret.shape[0]} x {ret.shape[1]}", end='\n\n')
    return ret

def poly_reg(X, Y, Xtest, Ytest, info, order):
    mylr = MyLinearRegression(thetas=info["theta"], alpha=info["alpha"], max_iter=info["iter"])
    X = polynomial_features(X, order)
    ret = mylr.fit_(X, Y)
    mse=MyLinearRegression.mse_(Ytest, mylr.predict_(Xtest))
    return (mylr.thetas, mse)

if __name__ == "__main__":

    test = np.array([[10., 20.], [30., 40.]])
    print(test)
    test = minmax(test)
    print(test)
    test = antiminmax(test)
    print (test)
    sys.exit()
    path_data='space_avocado.csv'
    df = load_data(path_data)
    if df is None:
        print(f"Error loading data from {path_data}")
        sys.exit()


    X = df[['weight', 'prod_distance', 'time_delivery']].to_numpy()
    Y = df[['target']].to_numpy()

    (Xtrain, Xtest, Ytrain, Ytest) = data_spliter(X, Y, 0.5)

    ret = poly_reg(Xtrain, Ytrain,  Xtest, Ytest, {"theta" : [[200000.], [4700.], [150.], [8000.]], "alpha" : 4e-7, "iter" : 2000000}, 1)

    print (ret)

