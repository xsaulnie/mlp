import numpy as np

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
            self.thetas = self.thetas - (self.alpha * MyLinearRegression.grad_(x, y, self.thetas))
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

if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])


    print("real value of y :")
    print(Y)
    print("Prediction before fit :")
    y_hat = mylr.predict_(X)
    print(y_hat)
    print("Loss element from each datapoint :")
    print(mylr.loss_elem_(Y, y_hat))
    print("Total loss by using a mean on loss elements :")
    print(mylr.loss_(Y, y_hat))
    print("fiting data for alpha=1.6e-4, max_iter=200000 ...")
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.fit_(X, Y)
    print("theta result from fiting :")
    print(mylr.thetas)
    print("prediction after fit :")
    y_hat = mylr.predict_(X)
    print(y_hat)
    print("real value of y :")
    print(Y)

    print("Loss elems and loss after fit :")
    print(mylr.loss_elem_(Y,y_hat))
    print("total loss : ", mylr.loss_(Y, y_hat))
    print("The linear regression seeem to be quite successfull :)")