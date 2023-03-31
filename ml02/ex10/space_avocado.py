from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as mat
import sys
from tqdm import tqdm
from pickle import *

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
    def r2score_(y, y_hat):
        if not MyLinearRegression.check_matix(y) or not MyLinearRegression.check_matix(y_hat):
            return (None)

        if y.shape[0] != y_hat.shape[0]:
            return (None)

        y_mean = sum(y) / y.shape[0]

        r2score = 1 - (np.sum((y_hat - y) * (y_hat - y)) / np.sum((y - y_mean) * (y - y_mean)))

        return (r2score)

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
    Y_hat = mylr.predict_(polynomial_features(Xtest, order))
    mse=MyLinearRegression.mse_(Ytest, Y_hat)
    r2 = MyLinearRegression.r2score_(Ytest, Y_hat)
    return (mylr.thetas, mse, r2)

if __name__ == "__main__":

    path_data='space_avocado.csv'
    df = load_data(path_data)
    if df is None:
        print(f"Error loading data from {path_data}")
        sys.exit()

    f = open("models.pickle", "rb")

    test = load(f)

    print(test)

    X = df[['weight', 'prod_distance', 'time_delivery']].to_numpy()
    Y = df[['target']].to_numpy()

    (Xtrain, Xtest, Ytrain, Ytest, Rtrain, Rtest) = data_spliter(X, Y, 0.5, normilize=True)

    #resultheta = poly_reg(Xtrain, Ytrain,  Xtest, Ytest, {"theta" : [[0.], [0.], [0.], [0.], [0.], [0,], [0.], [0.], [0.], [0.], [0.], [0.], [0.]], "alpha" : 1e-2, "iter" : 20000}, 4) #2000000}
    resultheta = test[4]
    X_prime = np.concatenate((np.ones((Xtest.shape[0], 1)), polynomial_features(Xtest, 4)), axis=1).astype(float)
    Y_hat =  (np.matmul(X_prime, resultheta[0]).astype(float))
    S_hat = Y_hat
    Y_hat = antiminmax(Rtrain[:, -1:], Y_hat)

    plt.subplot(2, 2, 1)
    plt.xlabel("Weight order (in tons)")
    plt.subplot(2, 2, 2)
    plt.xlabel("Produced distance (in Mkm)")
    plt.subplot(2, 2, 3)
    plt.xlabel("Delivery time (in days)")

    for it in range(3):
        plt.subplot(2, 2, it + 1)
        plt.scatter(Rtest[:, it], Rtest[:, -1:], s=8, label="Price")
        plt.scatter(Rtest[:, it], Y_hat, s=2, label="Pred")
        plt.legend(loc="upper center")
        plt.ylabel("Price (trantorian unit)")

    plt.suptitle("Space Avocado's Price Linear Polynomial regression of order 4\nMSE:%.8f\nR2:%.8f" % (resultheta[1], resultheta[2]))

    plt.subplot(2, 2, 4)
    xbar = np.arange(4)
    mse = [test[idx + 1][1] * 1000 for idx in range(4)]
    r2 = [test[idx + 1][2] for idx in range(4)]
    plt.bar(xbar, mse, color='b', width=0.3, edgecolor ='grey', label='mean square error')
    plt.bar(xbar + 0.3 , r2, color='r', width = 0.3, edgecolor ='grey', label='r2 score')
    plt.xlabel('order of the model')
    plt.ylabel('value of the metric')
    plt.legend()
    plt.xticks([r + 0.3 for r in range(4)], ['1', '2', '3', '4'])

    plt.show()


    