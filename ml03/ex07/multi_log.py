import numpy as np
import math as mat
import sys
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

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

        for it in tqdm(range(self.max_iter)):
            self.theta = self.theta - (self.alpha * MyLogisticRegression.grad(x, y, self.theta))
            if True in np.isnan(self.theta):
                return None
        return self.theta


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

def data_spliter(x, y, proportion, normilize=False):

    if proportion > 1 or proportion < 0:
        return None

    data = x

    nb = int(proportion * data.shape[0])

    train = data[0:nb, :]
    test = data[nb:, :]

    ytrain = y[0:nb, :]
    ytest = y[nb:, :]

    if normilize == True:
        (ntrain, ntest) = minmax(train, test)
        return((ntrain, ntest, ytrain, ytest, train, test))

    return((train, test, ytrain, ytest))


def parcing_arg(argv):
    if len(argv) == 1:
        print("usage:\tpython mono_log.py -zipcode=x\n\tx being 0, 1, 2 or 3")
        return -1
    if len(argv) > 2:
        print("Error: Wrong number of arguments")
        return -1
    kwarg={}
    sp = argv[1].split('=')
    if (len(sp) != 2):
        print("Error : Wrong arguments")
        return -1
    kwarg[sp[0]] = sp[1]

    if not "-zipcode" in kwarg.keys():
        print("Error : Wrong argument")
        return -1
    
    num = kwarg["-zipcode"]

    if (num != "0" and num != "1" and num != "2" and num != "3"):
        print("Error : Zipcode is not a correct number")
        return -1
    return(int(num))

def load_data(path):
    if not type(path) is str:
        return None
    try:
        ret = pd.read_csv(path)
    except:
        return None
    print(f"Loading dataset from {path} of dimensions {ret.shape[0]} x {ret.shape[1]}", end='\n\n')
    return ret

def planet_filter(Y, zipc):
    Y_ret = np.copy(Y)
    for it in range(Y.shape[0]):
        if Y_ret[it][0] == zipc:
            Y_ret[it][0] = 1.
        else:
            Y_ret[it][0] = 0.
    return(Y_ret)

def prediction_filter(y, rate):
    if (rate < 0 or rate > 1):
        return None
    ret = np.copy(y)
    for lin in range(y.shape[0]):
        if (y[lin][0] > rate):
            ret[lin][0] = 1.0
        else:
            ret[lin][0] = 0.0
    return ret

def correct_ratio(Y_hat, Ytest, zipc):
    correct = 0
    false_pos = 0
    missed = 0

    Y_test = planet_filter(Ytest, zipc)

    for idx in range(Y_hat.shape[0]):
        if (Y_hat[idx][0] == Y_test[idx][0]):
            correct = correct + 1
        else:
            if Y_hat[idx][0] == 1.:
                false_pos = false_pos + 1
            else:
                missed = missed + 1

    print("Result : %d correct estimations out of %d test" % (correct, Y_hat.shape[0]))
    print("Accurancy : %d/%d meaning %.4f %%, with %d false positiv and %d tests missed (false negativ)" % (correct, Y_hat.shape[0], correct/Y_hat.shape[0]*100, false_pos, missed))

def plot_true_data(x, y, true_data, pred_data, errors, label, stat):
    colors = {0: "pink", 1: "brown", 2 : "red", 3 : "black"}

    plt.subplot(1, 2, 1)
    plt.title("Citizen's classification of the test dataset")
    for idx in range(4):
        plt.scatter(true_data[idx][:, x], true_data[idx][:, y] , label=planet[idx], color=colors[idx])
    plt.legend()
    plt.xlabel(label[1])
    plt.ylabel(label[3])

    plt.subplot(1, 2, 2)
    plt.title("Classification prediction from the regression")
    for idx in range(4):
        plt.scatter(pred_data[idx][:, x], pred_data[idx][:, y], label=planet[idx], color=colors[idx])

    plt.xlabel(label[1])
    plt.ylabel(label[3])
    plt.scatter(errors[:, x], errors[:, y], label="errors",color="blue", marker='x', s = 15)
    plt.legend()

    plt.suptitle(f"Comparaison beetwin prediction and real data\nVisualisation displaying {label[0]} in respect of {label[2]}\nScore : {stat[0]}/{stat[1]} {stat[2]} %")
    plt.subplots_adjust(top=0.8)

    plt.show()

def correct_model(Y_res, Ytest):
    correct = 0
    for idx in range(Ytest.shape[0]):
        if (float(Y_res[idx]) == Ytest[idx][0]):
            correct = correct + 1
    print(f"Result : {correct} citizenship guessed correctly out of {Ytest.shape[0]} citizens of the test dataset", end=", ")
    print("Precision : %d/%d, %.4f %%" % (correct, Ytest.shape[0], correct/Ytest.shape[0] * 100))
    return ((correct, Ytest.shape[0], round(correct/Ytest.shape[0] * 100), 2))


if __name__ == '__main__':
    planets = {0 : "The flying cities of Venus", 1: " United Nations of Earth", 2 : "Mars Republic", 3 : "The Asteroids' Belt colonies"}
    planet = {0 : "Venus", 1 : "Earth", 2: "Mars", 3: "Asteroid"}


    path_data='solar_system_census.csv'
    path_pred='solar_system_census_planets.csv'
    df = load_data(path_data)
    if df is None:
        print(f"Error loading data from {path_data}")
        sys.exit()
    df_pred = load_data(path_pred)
    if df_pred is None:
        print(f"Error loading data from {path_pred}")
        sys.exit()

    Y = df_pred[["Origin"]].to_numpy()
    X = df[["weight", "height", "bone_density"]].to_numpy()

    (Xtrain, Xtest, Ytrain, Ytest, Rtrain, Rtest) = data_spliter(X, Y, 0.5, normilize=True)


    All_Y = []
    for planx in range(4):
        print(f"{planets[planx]} 's logistical regression alpha=1e-2, max_iteration=2000000 from null thetas, fitting data...")
        Yptrain = planet_filter(Ytrain, planx)
        mlr = MyLogisticRegression(np.zeros((4, 1)), max_iter=2000, alpha=1e-2)
        mlr.fit_(Xtrain, Yptrain)
        Y_hat = mlr.predict_(Xtest)
        All_Y.append(Y_hat)
        print("theta obtened : [[%.2f], [%.2f], [%.2f], [%.2f]]" % (mlr.theta[0][0], mlr.theta[1][0], mlr.theta[2][0], mlr.theta[3][0]), end=" ")
        print("with a loss of %.6f" % (mlr.loss_(planet_filter(Ytest, planx), Y_hat)))
        print(f"Logistical Regression {planet[planx]} vs All precision : ")
        correct_ratio(prediction_filter(Y_hat, 0.8), Ytest, planx)
        print()

    Y_res = []

    for lin in range(Xtest.shape[0]):
        maxi = All_Y[0][lin][0]
        residx = 0
        for idx in range(4):
            if (maxi < All_Y[idx][lin][0]):
                maxi = All_Y[idx][lin][0]
                residx = idx
        Y_res.append(residx)


    print("After the 4 ONE vs ALL Logistical regression, We predict each citizenship on the test dataset from our model.")
    stat = correct_model(Y_res, Ytest)

    true_data = []
    for planx in range(4):
        data = []
        for idx in range(Xtest.shape[0]):
            if (Ytest[idx][0] == float(planx)):
                data.append(Rtest[idx])
        true_data.append(np.array(data))

    pred_data = []
    for planx in range(4):
        data = []
        for idx in range(Xtest.shape[0]):
            if (Y_res[idx] == planx):
                data.append(Rtest[idx])
        pred_data.append(np.array(data))

    errors = []
    for planx in range(4):
        data = []
        for idx in range(Xtest.shape[0]):
            if (Ytest[idx][0] != float(Y_res[idx])):
                errors.append(Rtest[idx])

    errors = np.array(errors)

    plot_true_data(0, 1, true_data, pred_data, errors, ("Weight", "weight (in pounds)", "Height", "height (in inches)"), stat)
    plot_true_data(2, 1, true_data, pred_data, errors, ("Bone density", "bone density", "Height", "height (in inches)"), stat)
    plot_true_data(0, 2, true_data, pred_data, errors, ("Weight", "weight (in pounds)", "Bone density", "bone density"), stat)




    

