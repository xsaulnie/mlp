import numpy as np
import math as mat

class MyLogisticRegression():
    """
        Description:
        My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        if not MyLogisticRegression.check_matix(theta):
            return None
        if theta.shape[1] != 1:
            return None
        if not type(alpha) is float:
            return None
        if not type(max_iter) is int:
            return None

        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

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
    def sigmoid(x):
        sig = lambda x : 1/ (1 + mat.exp(-x))
        return (np.array([[sig(elem)] for elem in x]))

    @staticmethod
    def grad(x, y, theta):
        X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)
        return (np.matmul(X_prime.T, (MyLogisticRegression.sigmoid(np.matmul(X_prime, theta)) - y)) / x.shape[0])

    def predict_(self, x):
        if (not MyLogisticRegression.check_matix(x)):
            return (None)
        if x.shape[1] != self.theta.shape[0] - 1:
            return None
        X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)
        return (MyLogisticRegression.sigmoid(np.matmul(X_prime, self.theta)))

    def loss_elem_(self, y, yhat):
        if (not MyLogisticRegression.check_matix(y) or not MyLogisticRegression.check_matix(yhat)):
            return None
        if not y.shape[1] == 1 or not yhat.shape[1] == 1:
            return None
        if not y.shape[0] == yhat.shape[0]:
            return None
        ret = np.zeros((y.shape[0], 1))

        for idx in range(y.shape[0]):
            ret[idx] =  y[idx][0] * mat.log(yhat[idx][0] + 1e-15) + (1 - y[idx][0])* mat.log(1 - yhat[idx][0] + 1e-15)
        return ret

    def loss_(self, y, yhat):
        if (not MyLogisticRegression.check_matix(y) or not MyLogisticRegression.check_matix(yhat)):
            return None
        if y.shape[1] != 1 or yhat.shape[1] != 1:
            return None
        if not y.shape[0] == yhat.shape[0]:
            return None

        ret = 0
        for idx in range(y.shape[0]):
            ret = ret + ( y[idx][0] * mat.log(yhat[idx][0] + 1e-15) + (1 - y[idx][0])* mat.log(1 - yhat[idx][0] + 1e-15))
        return (- ret / y.shape[0])

    def fit_(self, x, y):
        if not MyLogisticRegression.check_matix(x) or not MyLogisticRegression.check_matix(y):
            return None
        if y.shape[1] != 1 or x.shape[0] != y.shape[0]:
            return None
        if x.shape[1] != self.theta.shape[0] - 1:
            return None
        if (y.shape[0] != x.shape[0]):
            return None

        for it in range(self.max_iter):
            self.theta = self.theta - (self.alpha * MyLogisticRegression.grad(x, y, self.theta))
            if True in np.isnan(self.theta):
                return None
        return self.theta


if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    mylr = MyLogisticRegression(thetas)
    print("real value of Y")
    print(Y)
    print("prediction before fit")
    Y_hat = mylr.predict_(X)
    print(Y_hat)

    print("loss before fit")
    print(mylr.loss_(Y, Y_hat))

    print("fitting..")
    mylr.fit_(X, Y)
    print("thetas resulting from fit :")
    print(mylr.theta)

    print("prediction after fit")
    Y_hat = mylr.predict_(X)
    print(Y_hat)

    print("loss after fit")
    print(mylr.loss_(Y, Y_hat))

