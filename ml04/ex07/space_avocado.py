from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as mat
import sys
from tqdm import tqdm
from pickle import *


def minmax_(vec, ref):
    xvec = np.copy(vec)

    amin = np.amin(ref[:, 0])
    amax = np.amax(ref[:, 0])

    for idx in range(vec.shape[0]):
        xvec[idx][0] = (xvec[idx][0] - amin) / (amax - amin)
    return xvec

def antiminmax(train, test):
    xtest = np.copy(test)

    for col in range(train.shape[1]):
        amin = np.amin(train[:, col])
        amax = np.amax(train[:, col])
        for lin in range(xtest.shape[0]):
            xtest[lin][col] = xtest[lin][col] * (amax - amin) + amin       

    return xtest


def minmax(train, cross, test):
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
    xcross = np.copy(cross)
    xtest = np.copy(test)


    for col in range(xtrain.shape[1]):
        amin = np.amin(xtrain[:, col])
        amax = np.amax(xtrain[:, col])
        for lin in range(xtrain.shape[0]):
            xtrain[lin][col] = (xtrain[lin][col] - amin) / (amax - amin)
        for lin2 in range(xcross.shape[0]):
            xcross[lin2][col] = (xcross[lin2][col] - amin) / (amax - amin)
        for lin3 in range(xtest.shape[0]):
            xtest[lin3][col] = (xtest[lin3][col] - amin) / (amax - amin)

    return (xtrain, xcross, xtest)

def add_polynomial_features(x, power):
    ret = []
    for lin in range(x.shape[0]):
        pows = []
        for col in range(x.shape[1] * power):
            pows.append(x[lin][col % x.shape[1]] ** (int(col / x.shape[1]) + 1))
        ret.append(pows)

    return (np.array(ret))

def data_spliter(x, y, proportion, normilize=False):

    if proportion > 1 or proportion < 0:
        return None

    data = x

    nb = int(proportion * data.shape[0])
    nb2 = int(((data.shape[0] - nb) * proportion) + nb)

    train = data[0:nb, :]
    cross = data[nb:nb2, :]
    test = data[nb2:, :]

    ytrain = y[0:nb, :]
    ycross = y[nb:nb2, :]
    ytest = y[nb2:,  :]

    if normilize == True:
        (ntrain, ncross, ntest) = minmax(train, cross, test)
        return((ntrain, ncross, ntest, ytrain, ycross, ytest, test))

    return((train, cross, test, ytrain, ycross, ytest))

class MyRidge():
    """
    Description:
        My personnal ridge regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        if (type(thetas) is list):
            try:
                thetas = np.array(thetas)
            except:
                print("error")
                return None

        if not (MyRidge.check_matix(thetas)):
            return None

        if thetas.shape[1] != 1:
            return None

        if (not type(alpha) is float):
            return None

        if not (type(max_iter) is int):
            return None
        if not type(lambda_) is float and not type(lambda_) is int:
            return None


        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(thetas)
        self.lambda_ = lambda_

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
    def grad_(x, y, theta, lambda_):
        X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)
        theta_prime = np.copy(theta).astype(float)
        theta_prime[0][0] = 0
        return ((np.matmul(X_prime.T, (np.matmul(X_prime, theta) - y)) + lambda_ * theta_prime) / x.shape[0])
    
    @staticmethod
    def l2(theta):
        theta_prime = np.copy(theta).astype(float)
        theta_prime[0][0] = 0
        return (np.sum(np.dot(theta_prime.T, theta_prime)))

    def get_params_(self):
        return ({"thetas" : self.thetas, "alpha" : self.alpha, "max_iter" : self.max_iter, "lambda_" : self.lambda_})

    def set_params_(self, **params):

        valid = ["thetas", "alpha", "max_iter", "lambda_"]
        for key in params:
            if not key in valid:
                return None
            if key == "thetas":
                if not type(params[key]) is list and not type(params[key]) is np.ndarray:
                    return None
                if (type(params[key]) is list):
                    try:
                        params[key] = np.array(params[key])
                    except:
                        return None
            elif not type(params[key]) is type(getattr(self, key)):
                return None 

        for (key, val) in params.items():
            setattr(self, key, val)
        return self

    def fit_(self, x, y):
        if not MyRidge.check_matix(x) or not MyRidge.check_matix(y):
            return None
        if (y.shape[1] != 1):
            return None
        if (x.shape[0] != y.shape[0]):
            return None
        if (x.shape[1] != self.thetas.shape[0] - 1):
            return None

        for it in tqdm(range(self.max_iter)):
            self.thetas = self.thetas - (self.alpha * MyRidge.grad_(x, y, self.thetas, self.lambda_))
            if True in np.isnan(self.thetas):
                return None
        return self.thetas

    def predict_(self, x):
        if not MyRidge.check_matix(x):
            return None

        if (x.shape[1] != self.thetas.shape[0] - 1):
            return None

        X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)
        return (np.matmul(X_prime, self.thetas).astype(float))

    def loss_elem_(self, y, y_hat):
        if not MyRidge.check_matix(y) or not MyRidge.check_matix(y_hat):
            return None
        if (y.shape[0] != y_hat.shape[0]):
            return None
        if (y.shape[1] != 1 or y_hat.shape[1] != 1):
            return None

        sup_rige = MyRidge.l2(self.thetas) * self.lambda_
        return(((y_hat - y) * (y_hat - y)) + sup_rige)

    def loss_(self, y, y_hat):
        if not MyRidge.check_matix(y) or not MyRidge.check_matix(y_hat):
            return None
        if (y.shape[0] != y_hat.shape[0]):
            return None
        if (y.shape[1] != 1 or y_hat.shape[1] != 1):
            return None

        return ((np.dot((y_hat - y).T, (y_hat - y)) + self.lambda_ * MyRidge.l2(self.thetas)) / (2 * y.shape[0]))[0][0]

    def mse_(y, y_hat):
        if not MyRidge.check_matix(y) or not MyRidge.check_matix(y_hat):
            return None
        if (y.shape[0] != y_hat.shape[0]):
            return None
        if (y.shape[1] != 1 or y_hat.shape[1] != 1):
            return None
        return((sum((y_hat - y) * (y_hat - y)) / (y.shape[0]))[0])

    def r2score_(y, y_hat):
        if not MyRidge.check_matix(y) or not MyRidge.check_matix(y_hat):
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

def polytrain(X, Y, Xcross, Ycross, order, lambda_):
    mri = MyRidge(thetas=np.zeros((order * X.shape[1] + 1, 1)), alpha=1e-2, max_iter=20000, lambda_=lambda_)
    Xpol = add_polynomial_features(X, order)
    ret = mri.fit_(Xpol, minmax_(Y, Y))

    Y_hat = mri.predict_(add_polynomial_features(Xcross, order))
    mse = MyRidge.mse_(minmax_(Ycross, Y), Y_hat)
    r2 = MyRidge.r2score_(minmax_(Ycross, Y), Y_hat)
    return (mri.thetas, mse, r2)

def train_best(X, Y, Xtest, Ytest, order, lambda_):
    mri = MyRidge(thetas=np.zeros((order * X.shape[1] + 1, 1)), alpha=1e-2, max_iter=2000000, lambda_=lambda_)
    Xpol = add_polynomial_features(X, order)
    ret = mri.fit_(Xpol, minmax_(Y, Y))
    Y_hat = mri.predict_(add_polynomial_features(Xtest, order))
    mse = MyRidge.mse_(minmax_(Ytest, Y), Y_hat)
    r2 = MyRidge.r2score_(minmax_(Ytest, Y), Y_hat)
    Y_hat = antiminmax(Ytest, Y_hat)
    return (mri.thetas, mse, r2, Y_hat)



if __name__ == "__main__":

    path_data='space_avocado.csv'
    df = load_data(path_data)
    if df is None:
        print(f"Error loading data from {path_data}")
        sys.exit()

    X = df[['weight', 'prod_distance', 'time_delivery']].to_numpy()
    Y = df[['target']].to_numpy()
    (Xtrain, Xcross, Xtest, Ytrain, Ycross, Ytest, Rtest) = data_spliter(X, Y, 0.5, normilize=True)
    print("Spliting data, training : %d x %d, coss validation : %d x %d, testing : %d x %d" % (Xtrain.shape[0], Xtrain.shape[1], Xcross.shape[0],  Xcross.shape[1], Xtest.shape[0], Xtest.shape[1]))

    f = open("models.pickle", "rb")
    models = load(f)
    lamblist = [0, 0.2, 0.4, 0.6, 0.8, 1]

    best = train_best(Xtrain, Ytrain, Xcross, Ycross, 4, 0.0)
    Y_hat = best[3]

    plt.subplot(2, 2, 1)
    plt.xlabel("Weight order (in tons)")
    plt.subplot(2, 2, 2)
    plt.xlabel("Produced distance (in Mkm)")
    plt.subplot(2, 2, 3)
    plt.xlabel("Delivery time (in days)")

    for it in range(3):
        plt.subplot(2, 2, it + 1)
        plt.scatter(Rtest[:, it], Ytest, s=8, label="Price")
        plt.scatter(Rtest[:, it], Y_hat, s=2, label="Pred")
        plt.legend(loc="upper center")
        plt.ylabel("Price (trantorian unit)")

    plt.show()



    # for order in range(1, 5):
    #     plt.subplot(2, 2, order)
    #     plt.xlabel("lambdas")
    #     plt.ylabel("mean square error")
    #     plt.title("Order %d" % (order))

    #     mse = []
    #     for idx in lamblist:
    #         mse.append(models[order][idx][1] * 10000)

    #     plt.plot(lamblist, mse, '-o')

    # plt.suptitle(f"Ridge Regression mean square error\nDepending on lambda and order")

    # plt.show()

    # for order in range(1, 5):
    #     plt.subplot(2, 2, order)
    #     plt.xlabel("lambdas")
    #     plt.ylabel("r2 score")
    #     plt.title("Order %d" % (order))

    #     r2 = []
    #     for idx in lamblist:
    #         r2.append(models[order][idx][2])

    #     plt.plot(lamblist, r2, '-o', color='red')

    # plt.suptitle(f"Ridge Regression r2Score\nDepending on lambda and order")

    # plt.show()

    

    


    