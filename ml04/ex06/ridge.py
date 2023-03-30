import numpy as np

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
    def grad_(x, y, theta):
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
            if key is "thetas":
                if not type(params[key]) is list and not type(params[key]) is np.ndarray:
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

        for it in range(self.max_iter):
            self.thetas = self.thetas - (self.alpha * MyRidge.grad_(x, y, self.thetas))
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

        sup_rige = MyRidge.l2(self.thetas) / (2 * y.shape[0])
        return(((y_hat - y) * (y_hat - y)) + sup_rige)

    def loss_(self, y, y_hat):
        if not MyRidge.check_matix(y) or not MyRidge.check_matix(y_hat):
            return None
        if (y.shape[0] != y_hat.shape[0]):
            return None
        if (y.shape[1] != 1 or y_hat.shape[1] != 1):
            return None

        return ((np.dot((y_hat - y).T, (y_hat - y)) + self.lambda_ * MyRidge.l2(self.thetas)) / (2 * y.shape[0]))[0][0]

if __name__ == "__main__":
    mri = MyRidge([[0.],[0.]])
    print(mri.get_params_())
    mri.set_params_(alpha=8.6, thetas=[[1.0], [1.0]])
    print(mri.get_params_())