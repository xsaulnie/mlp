import numpy as np
import math as mat

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
        return((sum((y_hat - y) * (y_hat - y)) / (2 * y.shape[0]))[0])
        
if __name__ == "__main__":
    print("Main exemple")
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])

    lr1 = MyLinearRegression(np.array([[2], [0.7]]))
    y_hat = lr1.predict_(x)

    print("prediction with no fit :")
    print(y_hat)

    print("loss_elem with no fit : ")
    print(lr1.loss_elem_(y, y_hat))

    print("loss : ", lr1.loss_(y, y_hat))


    lr2 = MyLinearRegression(np.array([[1], [1]]), 5e-8, 1500000)

    print("fiting with theta = (1, 1) alpha = 5e-8 max_iter = 1500000 ....")
    lr2.fit_(x, y)

    print("thetas after fit : ")
    print(lr2.thetas)

    print("prediction with the fit : ")
    y_hat = lr2.predict_(x)
    print(y_hat)

    print("loss elem with the fit : ")
    print(lr2.loss_elem_(y, y_hat))

    print("loss : ", lr2.loss_(y, y_hat))
    



