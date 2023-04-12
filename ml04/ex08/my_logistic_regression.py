import numpy as np
import math as mat
from tqdm import tqdm

class MyLogisticRegression():
    """
        Description:
        My personnal logistic regression to classify things.
    """
    supported_penalities = ['l2']
    def __init__(self, theta, alpha=0.001, max_iter=1000, penalty='l2', lambda_=1.0):
        if not MyLogisticRegression.check_matix(theta):
            return None
        if theta.shape[1] != 1:
            return None
        if not type(alpha) is float:
            return None
        if not type(max_iter) is int:
            return None
        if not type(lambda_) is float:
            return None
        if not type(penalty) is str and not penalty is None:
            return None

        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        self.penalty = penalty
        self.lambda_ = lambda_ if penalty in self.supported_penalities else 0.0

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
    def grad(x, y, theta, penalty, lambda_):
        X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)
        theta_prime = np.copy(theta).astype(float)
        theta_prime[0][0] = 0
        if (penalty == 'l2'):
            return ((np.matmul(X_prime.T, (MyLogisticRegression.sigmoid(np.matmul(X_prime, theta)) - y)) + lambda_ * theta_prime) / x.shape[0])
        if penalty is None:
            return (np.matmul(X_prime.T, (MyLogisticRegression.sigmoid(np.matmul(X_prime, theta)) - y))  / x.shape[0])

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
            ret[idx] =  y[idx][0] * mat.log(yhat[idx][0] + 1e-15) + (1 - y[idx][0]) * mat.log(1 - yhat[idx][0] + 1e-15)

    def loss_(self, y, yhat):
        if (not MyLogisticRegression.check_matix(y) or not MyLogisticRegression.check_matix(yhat)):
            return None
        if y.shape[1] != 1 or yhat.shape[1] != 1:
            return None
        if not y.shape[0] == yhat.shape[0]:
            return None

        ret = 0
        for idx in range(y.shape[0]):
            ret = ret + ( y[idx][0] * mat.log(yhat[idx][0] + 1e-15) + (1 - y[idx][0]) * mat.log(1 - yhat[idx][0] + 1e-15))
        if self.penalty == 'l2':
            theta_prime = np.copy(self.theta)
            theta_prime[0][0] = 0
            dot = np.sum(np.dot(theta_prime.T, theta_prime))
            return (- ret / y.shape[0] + (self.lambda_ / 2 * y.shape[0]) * dot)
        if self.penalty is None:
            return (-ret / y.shape[0])

    def fit_(self, x, y):
        if not MyLogisticRegression.check_matix(x) or not MyLogisticRegression.check_matix(y):
            return None
        if y.shape[1] != 1 or x.shape[0] != y.shape[0]:
            return None
        if x.shape[1] != self.theta.shape[0] - 1:
            return None
        if (y.shape[0] != x.shape[0]):
            return None

        for it in tqdm(range(self.max_iter)):
            self.theta = self.theta - (self.alpha * MyLogisticRegression.grad(x, y, self.theta, self.penalty, self.lambda_))
            if True in np.isnan(self.theta):
                return None
        return self.theta

if __name__ == "__main__":
    print("Exemple from main\n")
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    print("Exemple 1\n")
    model1 = MyLogisticRegression(theta, lambda_=5.0)
    print (model1.penalty)

    print(model1.lambda_)
    print("Exemple 2\n")
    model2 = MyLogisticRegression(theta, penalty=None)
    print(model2.penalty)
    print(model2.lambda_)
    print("Exemple 3\n")
    model3 = MyLogisticRegression(theta, penalty=None, lambda_=2.0)
    print(model3.penalty)
    print(model3.lambda_)

    print("Logistic regression exemple with lambda=0\n")
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    mylr = MyLogisticRegression(thetas, lambda_= 0.0)
    print("prediction before fit")
    Y_hat = mylr.predict_(X)
    print(Y_hat)
    print("loss before fit")
    print(mylr.loss_(Y, Y_hat))
    print("fitting..")
    mylr.fit_(X, Y)
    print(mylr.theta)
    print("prediction after fit")
    Y_hat = mylr.predict_(X)
    print(Y_hat)
    print("loss after fit")
    print(mylr.loss_(Y, Y_hat))

    print("Logistic regression exemple with lambda=0.5\n")
    mylr2 = MyLogisticRegression(thetas, lambda_=0.5)
    print("prediction before fit")
    Y_hat = mylr2.predict_(X)
    print(Y_hat)
    print("loss before fit")
    print(mylr2.loss_(Y, Y_hat))
    print("fitting..")
    mylr2.fit_(X, Y)
    print(mylr.theta)
    print("prediction after fit")
    Y_hat = mylr2.predict_(X)
    print(Y_hat)
    print("loss after fit")
    print(mylr2.loss_(Y, Y_hat))






