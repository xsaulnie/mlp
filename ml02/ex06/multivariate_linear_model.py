from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as mat
import sys

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

        for it in range(self.max_iter):
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
    print(f"Loading dataset of dimensions {ret.shape[0]} x {ret.shape[1]}")
    return ret

if __name__ == "__main__":
    path_data = "spacecraft_data.csv"
    df = load_data(path_data)
    if df is None:
        print(f"Error loading data from {path_data}")
        sys.exit()

    print("Linear Regression of Price with Age as feature")

    X = df[['Age']].to_numpy()
    Y = df[['Sell_price']].to_numpy()

    mylr_age= MyLinearRegression(thetas=[[500.], [0.]], alpha = 2.5e-3, max_iter = 10000)
    print("Mean square error before fiting : %f" % (MyLinearRegression.mse_(Y, mylr_age.predict_(X))))
    mylr_age.fit_(X, Y)
    
    age_pred = mylr_age.predict_(X)
    mse = MyLinearRegression.mse_(Y, age_pred)
    print("Mean square error after fiting : %f" % (mse))
    print(f"Regression : Price = {mylr_age.thetas[1][0]} * Age + {mylr_age.thetas[0][0]}")

    fig = plt.figure()
    plt.title("Age Univariate Linerar Regression\n Price = %.2f * Age + %.2f\nMSE : %.0f " % (mylr_age.thetas[1][0], mylr_age.thetas[0][0], mse))
    plt.scatter(X, Y, color="midnightblue", label="Sell price")
    plt.scatter(X, age_pred, color="deepskyblue", label="Predicted sell price", s=20)
    plt.xlabel("x : age (in years)")
    plt.ylabel("y : sell price (in keuros)")
    plt.legend(loc="lower right")
    fig.subplots_adjust(bottom=0.125)
    fig.subplots_adjust(top=0.85)
    fig.text(0.25, 0.005, "thetas=[500, 0] alpha=2.5e-3 iter=500000")
    plt.show()
    print("\n\n")
    
    