from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as mat
import sys


class FileLoader:
    def load(self, path):
        if (not type(path) is str):
            return None
        try:
            ret = pd.read_csv(path)
        except:
            return None
        print(f"Loading dataset from '{path}' of dimensions {ret.shape[0]} x {ret.shape[1]}")
        return(ret)
    def display(self, df, n):
        if (not isinstance(df, pd.DataFrame) or not type(n) is int):
            return
        if (n > 0):
            print (df.loc[:n - 1])
        else:
            print(df.loc[df.shape[0] + n:])

class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        if not (MyLinearRegression.vec_col(thetas)):
            return None
        if thetas.shape[0] != 2:
            return None
        if (not type(alpha) is float):
            return None
        if not (type(max_iter) is int):
            return None
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    @staticmethod
    def vec_col(vec):
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

    @staticmethod
    def simple_gradient(x, y, theta):
        X_prime = np.concatenate((np.ones(x.shape[0], dtype=x.dtype).reshape(x.shape[0], 1), x), axis=1)
        grad = np.matmul(X_prime.T, np.matmul(X_prime, theta).astype(float) - y) / y.shape[0]
        return grad

    def fit_(self, x, y):
        if not MyLinearRegression.vec_col(x) or not MyLinearRegression.vec_col(y):
            return None
        new_theta = np.copy(self.thetas).astype(float)
        for it in range(self.max_iter):
            grad = MyLinearRegression.simple_gradient(x, y, new_theta)
            new_theta[0][0] = new_theta[0][0] - (self.alpha * grad[0][0])
            if mat.isnan(new_theta[0][0]):
                return None
            new_theta[1][0] = new_theta[1][0] - (self.alpha * grad[1][0])
            if mat.isnan(new_theta[1][0]):
                return None
        self.thetas = new_theta
        return new_theta

    def predict_(self, x):
        if not MyLinearRegression.vec_col(x):
            return None
        X_prime = np.concatenate((np.ones(x.shape[0], dtype=x.dtype).reshape(x.shape[0], 1), x), axis=1)
        return (np.matmul(X_prime, self.thetas).astype(float))

    def loss_elem_(self, y, y_hat):
        if not MyLinearRegression.vec_col(y) or not MyLinearRegression.vec_col(y_hat):
            return None
        if (y.shape[0] != y_hat.shape[0]):
            return None
        return((y_hat - y) * (y_hat - y))

    def loss_(self, y, y_hat):
        if not MyLinearRegression.vec_col(y) or not MyLinearRegression.vec_col(y_hat):
            return None
        if (y.shape[0] != y_hat.shape[0]):
            return None
        return(sum((y_hat - y) * (y_hat - y)) / (2 * y.shape[0]))

    def mse_(y, y_hat):
        if not MyLinearRegression.vec_col(y) or not MyLinearRegression.vec_col(y_hat):
            return None
        if (y.shape[0] != y_hat.shape[0]):
            return None
        return((sum((y_hat - y) * (y_hat - y)) / (y.shape[0]))[0])

    def plot_regression(self, x, y):
        fig = plt.figure()
        y_hat = self.predict_(x)
        mse = MyLinearRegression.mse_(y, y_hat) 
        plt.title(f"Linear Regression, theta {self.thetas}, mse {mse}")
        plt.scatter(x, y, color='blue', label='Strue')

        plt.scatter(x, y_hat, color = 'green', label='Spredict')

        xplot = np.linspace(np.amin(x), np.amax(x), 100)
        yplot = xplot * self.thetas[1][0] + self.thetas[0][0]
        plt.plot(xplot, yplot, '--', color='green', label='Reg')
        plt.legend(loc="upper center", ncol=3, frameon=False)
        plt.xlabel("Quantity of blue pill (in micrograms)")
        plt.ylabel("Space driving score")

        plt.show()
    def plot_loss_function(self, x, y):

        fig = plt.figure()

        save = self.thetas[0][0]
        self.thetas[0][0] = int(self.thetas[0][0]) - 6

        def Jthet(t1):
            theta= np.array([[self.thetas[0][0]], [t1]])
            X_prime = np.concatenate((np.ones(x.shape[0], dtype=x.dtype).reshape(x.shape[0], 1), x), axis=1)
            y_pred = np.matmul(X_prime, theta).astype(float)
            return((sum((y_pred - y) * (y_pred - y)) / (2 * y.shape[0]))[0])

        for idx in range(6):
            xplot = np.linspace(-14, -4, 100)
            vfunc = np.vectorize(Jthet)
            yplot =  vfunc(xplot)
            plt.plot(xplot, yplot, label=f'θ0={self.thetas[0][0]}')
            self.thetas[0][0] = self.thetas[0][0] + 2

        self.thetas[0][0] = save

        plt.xlabel("Theta1")
        plt.ylabel("Cost function J")
        plt.legend(loc="lower right", frameon=False)
    
        plt.show()

if __name__ == "__main__":
    loader = FileLoader()
    data = loader.load("are_blue_pills_magics.csv")
    if (data is None):
        print("Error loading data")
        sys.exit()


    Xpill = np.array(data["Micrograms"]).reshape(-1,1)
    Yscore = np.array(data["Score"]).reshape(-1, 1)

    linear_model = MyLinearRegression(np.array([[89.0], [-8]]))

    linear_model.fit_(Xpill, Yscore)
    linear_model.plot_loss_function(Xpill, Yscore)
    sys.exit()


    Y_model = linear_model.predict_(Xpill)
    print("mse :" , MyLinearRegression.mse_(Yscore, Y_model))
    linear_model.plot_regression(Xpill, Yscore)


    linear_model.fit_(Xpill, Yscore)
    print("mse :" , MyLinearRegression.mse_(Yscore, linear_model.predict_(Xpill)))

    linear_model.plot_regression(Xpill, Yscore)