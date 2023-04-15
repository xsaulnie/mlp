from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as mat
import sys
from tqdm import tqdm

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

def plot_univariate(X, Y, thetas, ft, info):
    print(f"Linear Regression of Price with {ft} as feature")
    mylr= MyLinearRegression(thetas, alpha = info[ft]["alpha"], max_iter = info[ft]["iter"])
    mse_begin = MyLinearRegression.mse_(Y, mylr.predict_(X))
    print("Mean square error before fiting : %f" % (mse_begin))
    print("alpha=%.4f, iterations=%d Fit computation..." % (info[ft]["alpha"], info[ft]["iter"]))
    mylr.fit_(X, Y)

    pred = mylr.predict_(X)
    mse = MyLinearRegression.mse_(Y, pred)
    print("Mean square error after fiting : %f, gained : %s" % (mse, mse_begin - mse))
    print(f"Regression : Price = {mylr.thetas[1][0]} * {ft} + {mylr.thetas[0][0]}", end="\n\n")

    fig = plt.figure()
    plt.title("%s Univariate Linerar Regression\n Price = %.2f * %s + %.2f\nMSE : %.0f " % (ft, mylr.thetas[1][0], ft, mylr.thetas[0][0], mse))
    plt.scatter(X, Y, color=info[ft]["cdata"], label="Sell price")
    plt.scatter(X, pred, color=info[ft]["cpred"], label="Predicted sell price", s=15)
    plt.xlabel(info[ft]["label"])
    plt.ylabel("y : sell price (in keuros)")
    plt.legend(loc="lower right")
    fig.subplots_adjust(bottom=0.125)
    fig.subplots_adjust(top=0.85)
    al = info[ft]["alpha"]
    it = info[ft]["iter"]
    fig.text(0.01, 0.005, f"t0=[{thetas[0][0]}, {thetas[1][0]}] alpha={al}, iter={it}", fontsize=8)
    plt.grid()
    plt.show()

def plot_multivariate(thetas, ft, info, mse):
    fig = plt.figure()
    Y_pred = mylr.predict_(X)

    if (ft == 'Age'):
        nb = 0
    elif (ft == 'Thrust_power'):
        nb = 1
    elif (ft == 'Terameters'):
        nb = 2
    else:
        return None

    plt.title("Multivariate Linear Regression\n Price in respect to %s\nMSE: %.0f" % (ft, mse))
    plt.scatter(X[:, nb], Y, color=info[ft]["cdata"], label="Sell price")
    plt.scatter(X[:, nb], Y_pred, color=info[ft]["cpred"], label="predicted sell price", s=15)
    plt.xlabel(info[ft]["label"])
    plt.ylabel("y : sell price (in keuros)")
    plt.legend(loc="lower right")
    fig.subplots_adjust(bottom=0.125)
    fig.subplots_adjust(top=0.85)
    fig.text(0.01, 0.005, "Price = %.2f * Age + %.2f * Thrust_power + %.2f * Terameters + %.2f" % (thetas[1][0], thetas[2][0], thetas[3][0], thetas[0][0]), fontsize=8)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    path_data = "spacecraft_data.csv"
    df = load_data(path_data)
    if df is None:
        print(f"Error loading data from {path_data}")
        sys.exit()

    Y = df[['Sell_price']].to_numpy()

    info = {"Age" : {"cdata" : "midnightblue", "cpred" : "deepskyblue", "label" : "x1: age (in years)", "alpha": 2.5e-3, "iter": 500000 }}
    info.update({"Thrust_power" : {"cdata" : "g", "cpred" : "chartreuse", "label" : "x2: thrust power(in 10Km/s)", "alpha": 2.5e-5 , "iter" : 500000}})
    info.update({"Terameters" : {"cdata" : "darkviolet", "cpred" : "violet", "label" : "x3 : distance totalizer value of spacecraft (in Tmeters)", "alpha": 2.5e-5 , "iter" : 500000}})  


    # plot_univariate(df[['Age']].to_numpy(), Y, [[500.], [0.]], "Age", info)
    # plot_univariate(df[['Thrust_power']].to_numpy(), Y, [[0.], [4.]], "Thrust_power", info)
    # plot_univariate(df[['Terameters']].to_numpy(), Y, [[700.],[-2.]], "Terameters", info)


    print("Multivariate linear regression of the Price in respect to the Age, the Thruse_power and the Terameters")
    X = df[['Age', 'Thrust_power', 'Terameters']].to_numpy()

    mylr = MyLinearRegression(thetas = [[350.], [-20.0], [5.0], [-2.0]], alpha = 5e-5, max_iter = 500000)
    print("Mean Square Error before fitting : ", MyLinearRegression.mse_(Y, mylr.predict_(X)))
    print("Fit computation...")
    mylr.fit_(X, Y)
    mse =  MyLinearRegression.mse_(Y, mylr.predict_(X))
    print("Mean Square Error after fitting : ", mse)
    print(f"Regression : Price = {mylr.thetas[1][0]} * Age + {mylr.thetas[2][0]} * Thrust_power + {mylr.thetas[3][0]} * Terameters + {mylr.thetas[0][0]}", end="\n\n")
    print("Mean Square Error obtained is lower than each univariate regressions.")
    print("The multivariate linear regression is way more successful than the univariate !")
    print("Indeed it took into account each features of the dataset ;)")

    plot_multivariate(mylr.thetas, "Age", info, mse)
    plot_multivariate(mylr.thetas, 'Thrust_power', info, mse)
    plot_multivariate(mylr.thetas, "Terameters", info, mse)



    