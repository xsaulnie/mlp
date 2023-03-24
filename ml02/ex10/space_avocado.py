from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as mat
import sys
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

def minmax(train, test):
    if not (isinstance(train, np.ndarray) or not isinstance(test, np.ndarray)):
        return None
    if (train.dtype != "float64" and train.dtype != "int64"):
        return None
    if (test.dtype != "float64" and test.dtype != "int64"):
        return None
    if (len(train.shape) != 2 or len(test.shape) != 2):
        return None
    if (train.size == 0 or test.size == 0):
        return None

    xtrain = np.copy(train)
    xtest = np.copy(test)

    for col in range(xtrain.shape[1]):
        amin = np.amin(xtrain[:, col])
        amax = np.amax(xtrain[:, col])
        for lin in range(xtrain.shape[0]):
            xtrain[lin][col] = (xtrain[lin][col] - amin) / (amax - amin)
        for lin2 in range(xtest.shape[0]):
            xtest[lin2][col] = (xtest[lin2][col] - amin) / (amax - amin)       

    return (xtrain, xtest)


def antiminmax(train, test):
    xtest = np.copy(test)

    for col in range(train.shape[1]):
        amin = np.amin(train[:, col])
        amax = np.amax(train[:, col])
        for lin in range(xtest.shape[0]):
            xtest[lin][col] = xtest[lin][col] * (amax - amin) + amin       

    return xtest

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

def data_spliter(x, y, proportion, normilize=False):

    data = np.concatenate((x, y), axis = 1)
    np.random.shuffle(data)

    nb = int(proportion * data.shape[0])

    train = data[0:nb, :]
    test = data[nb:, :]

    if normilize == True:
        (ntrain, ntest) = minmax(train, test)
    return((ntrain[:,:-1], ntest[:,:-1], ntrain[:,-1:], ntest[:,-1:], train, test))

    return((train[:,:-1], test[:,:-1], train[:,-1:], test[:,-1:]))

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
    #print(mylr.predict_(Xtest))
    mse=MyLinearRegression.mse_(Ytest, mylr.predict_(polynomial_features(Xtest, order)))
    return (mylr.thetas, mse)

if __name__ == "__main__":

    path_data='space_avocado.csv'
    df = load_data(path_data)
    if df is None:
        print(f"Error loading data from {path_data}")
        sys.exit()


    X = df[['weight', 'prod_distance', 'time_delivery']].to_numpy()
    Y = df[['target']].to_numpy()

    (Xtrain, Xtest, Ytrain, Ytest, Rtrain, Rtest) = data_spliter(X, Y, 0.5, normilize=True)

    #ret = poly_reg(Xtrain, Ytrain,  Xtest, Ytest, {"theta" : [[200000.], [4700.], [150.], [8000.]], "alpha" : 4e-7, "iter" : 2000000}, 1)
    #ret = poly_reg(Xtrain, Ytrain,  Xtest, Ytest, {"theta" : [[0.1], [0.6], [0.], [0.]], "alpha" : 1e-3, "iter" : 2000000}, 1)
    #ret = poly_reg(Xtrain, Ytrain,  Xtest, Ytest, {"theta" : [[0.], [0.], [0.], [0.], [0.], [0,], [0.]], "alpha" : 1e-3, "iter" : 2000000}, 2)
    #ret = poly_reg(Xtrain, Ytrain,  Xtest, Ytest, {"theta" : [[0.], [0.], [0.], [0.], [0.], [0,], [0.], [0.], [0.], [0.]], "alpha" : 1e-3, "iter" : 2000000}, 3)
    #ret = poly_reg(Xtrain, Ytrain,  Xtest, Ytest, {"theta" : [[0.], [0.], [0.], [0.], [0.], [0,], [0.], [0.], [0.], [0.], [0.], [0.], [0.]], "alpha" : 1e-2, "iter" : 2000000}, 4)
    # print (ret)


    resultheta = np.array([[ 1.85634762e-01],
       [ 9.03278408e-01],
       [-7.62614727e-01],
       [-9.11332548e-03],
       [-3.65514205e-01],
       [ 1.24867537e+00],
       [ 6.82091855e-03],
       [ 2.34914225e-01],
       [-5.96444812e-02],
       [ 1.12623643e-03],
       [-5.79239440e-02],
       [-3.35464960e-01],
       [-3.21080547e-03]])

    X_prime = np.concatenate((np.ones((Xtest.shape[0], 1)), polynomial_features(Xtest, 4)), axis=1).astype(float)
    print(X_prime.shape)
    print(resultheta.shape)
    Y_hat =  (np.matmul(X_prime, resultheta).astype(float))
    print(Y_hat.shape)
    print(Rtrain.shape)
    Y_hat = antiminmax(Rtrain[:, -1:], Y_hat)

    plt.scatter(Rtest[:, 1], Rtest[:, -1:], s = 10)
    plt.scatter(Rtest[:, 1], Y_hat, s = 3)
    plt.show()

    