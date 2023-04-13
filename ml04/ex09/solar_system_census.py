import numpy as np
import math as mat
from tqdm import tqdm
import pandas as pd
import sys
from matplotlib import pyplot as plt
from pickle import *

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

def check_matix(mat):
    if not (isinstance(mat, np.ndarray)):
        return False
    if mat.dtype != "int64" and mat.dtype != "float64" and not (mat.dtype == 'U8'):
        return False
    if len(mat.shape) == 1:
        mat = np.atleast_2d(mat).T
    if len(mat.shape) != 2:
        return False
    if (mat.size == 0):
        return False
    return True

def load_data(path):
    if not type(path) is str:
        return None
    try:
        ret = pd.read_csv(path)
    except:
        return None
    print(f"Loading dataset from {path} of dimensions {ret.shape[0]} x {ret.shape[1]}", end='\n\n')
    return ret

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
        return((ntrain, ncross, ntest, ytrain, ycross, ytest, cross, test))

    return((train, cross, test, ytrain, ycross, ytest))

def add_polynomial_features(x, power):
    ret = []
    for lin in range(x.shape[0]):
        pows = []
        for col in range(x.shape[1] * power):
            pows.append(x[lin][col % x.shape[1]] ** (int(col / x.shape[1]) + 1))
        ret.append(pows)

    return (np.array(ret))

def f1_score_(y, y_hat, pos_label=1):
    if not check_matix(y) or not check_matix(y_hat):
        return None
    if len(y.shape) == 1:
        y = np.atleast_2d(y).T
    if len(y_hat.shape) == 1:
        y_hat = np.atleast_2d(y_hat).T
    if y.shape[1] != 1 or y_hat.shape[1] != 1:
        return None
    if (y.shape[0] != y_hat.shape[0]):
        return None
    if not type(pos_label) is str and not type(pos_label) is float and not type(pos_label) is int:
        return None

    st = {'tp' : 0, 'tn' : 0, 'fp' : 0, 'fn' : 0}

    for idx in range(y.shape[0]):
        if (y_hat[idx] == y[idx]):
            if y[idx] == pos_label:
                st['tp'] = st['tp'] + 1
            else:
                st['tn'] = st['tn'] + 1
        else:
            if y[idx] == pos_label:
                st['fn'] = st['fn'] + 1
            else:
                st['fp'] = st['fp'] + 1

    #print(st)
    if (st['tp'] == 0):
        return float(0)

    prec = st['tp'] / (st['tp'] + st['fp'])
    reca = st['tp'] / (st['tp'] + st['fn'])


    if (prec == 0 or reca == 0):
        return float(0)
    if (prec + reca == 0):
        return float('inf')

    return ((2 * prec * reca) / (prec + reca))

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

def train_best(lamb, planets, planet, Xtrain, Ytrain, Xtest, Ytest):
    ret = []
    info = []
    for curplan in range(4):
        Ytrain1 = planet_filter(Ytrain, curplan)
        mlr = MyLogisticRegression(np.array([[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]),max_iter=100000, alpha=1e-2, penalty='l2', lambda_=lamb)
        print(f"{planets[curplan]} vs All 's logistical regression alpha=1e-2, lambda={lamb}, max_iteration=1000000 from null thetas, fitting data...")
        Xpoly = add_polynomial_features(Xtrain, 3)
        mlr.fit_(Xpoly, Ytrain1)
        Yhat = mlr.predict_(add_polynomial_features(Xtest, 3))
        info.append(Yhat)
        Y_hat = prediction_filter(Yhat, 0.5)
        ret.append(Y_hat)
        f1sc = f1_score_(planet_filter(Ytest, curplan), Y_hat, pos_label=1.)
        print("theta obtened : [[%.2f], [%.2f], [%.2f], [%.2f]]" % (mlr.theta[0][0], mlr.theta[1][0], mlr.theta[2][0], mlr.theta[3][0]))
        print("with a loss of %.6f" % (mlr.loss_(planet_filter(Ytest, curplan), Yhat)))
        print("f1 score, planet %s, lambda %.1f : %f" % (planet[curplan], lamb, f1sc))

    Y_res = []
    for lin in range(Xtest.shape[0]):
        maxi = info[0][lin][0]
        residx = 0
        for idx in range(4):
            if (maxi < info[idx][lin][0]):
                maxi = info[idx][lin][0]
                residx = idx
        Y_res.append(residx)

    return(ret, np.array(Y_res).reshape(-1, 1))

def correct_ratio(Y_hat, Ytest, plan):
    correct = 0
    false_pos = 0
    missed = 0

    Yt = planet_filter(Ytest, plan)

    for idx in range(Y_hat.shape[0]):
        if (Y_hat[idx][0] == Yt[idx][0]):
            correct = correct + 1
        else:
            if Y_hat[idx][0] == 1.:
                false_pos = false_pos + 1
            else:
                missed = missed + 1
    return((correct, Y_hat.shape[0], correct / Y_hat.shape[0] * 100, false_pos, missed))

def correct_model(Y_res, Ytest):
    correct = 0
    for idx in range(Ytest.shape[0]):
        if (float(Y_res[idx]) == Ytest[idx][0]):
            correct = correct + 1
    print(f"Result : {correct} citizenship guessed correctly out of {Ytest.shape[0]} citizens of the test dataset", end=", ")
    print("Precision : %d/%d, %.4f %%" % (correct, Ytest.shape[0], correct/Ytest.shape[0] * 100))
    return ((correct, Ytest.shape[0], round(correct/Ytest.shape[0] * 100), 4))

if __name__ == "__main__":
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

    planets = {0 : "The flying cities of Venus", 1: " United Nations of Earth", 2 : "Mars Republic", 3 : "The Asteroids' Belt colonies"}
    planet = {0 : "Venus", 1 : "Earth", 2: "Mars", 3: "Asteroid"}

    Y = df_pred[["Origin"]].to_numpy()
    X = df[["weight", "height", "bone_density"]].to_numpy()

    (Xtrain, Xcross, Xtest, Ytrain, Ycross, Ytest, cross, test) = data_spliter(X, Y, 0.5, normilize=True)
    print("Spliting data, training : %d x %d, cross validation : %d x %d, testing : %d x %d" % (Xtrain.shape[0], Xtrain.shape[1], Xcross.shape[0],  Xcross.shape[1], Xtest.shape[0], Xtest.shape[1]), end='\n\n')

    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    f = open("models.pickle", "rb")
    modelr = load(f)

    print("Evaluating all model on test dataset :")
    for curplan in range(4):

        plt.subplot(2, 2, curplan + 1)
        plt.xlabel("lambdas")
        plt.ylabel("f1 score")

        f1p = []
        for lamb in lambdas:
            f1p.append(modelr[curplan][lamb][1])
            print(f"f1 score of model lambda={lamb}, {planet[curplan]} vs ALL : {modelr[curplan][lamb][1]}")
        plt.bar(lambdas, f1p, width = 0.05)
        plt.title(f"{planet[curplan]} vs ALL logistical regression\nMax: lambda={lambdas[f1p.index(max(f1p))]}")
    plt.suptitle("Displaying f1 score on the cross dataset \nfor various lambdas of the logistical regression")
    plt.show()



    ret = train_best(0.0, planets, planet, Xtrain, Ytrain, Xtest, Ytest)


    f = open("best.pickle", "wb")
    dump(ret, f)
    f.close()

    for curplan in range(4):
        plt.subplot(2, 2, curplan + 1)

        plan = []
        other = []
        detected = []
        for idx in range(test.shape[0]):
            if (Ytest[idx][0] == float(curplan)):
                plan.append(test[idx])
            else:
                other.append(test[idx])

        for idx in range(Ytest.shape[0]):
            if (ret[0][curplan][idx][0] == 1.):
                detected.append(test[idx])
        plan = np.array(plan)
        other = np.array(other)
        detected = np.array(detected)

        plt.scatter(plan[:, 0], plan[:, 1], label=planets[curplan], color="blue")
        plt.scatter(other[:, 0], other[:, 1], label="Other citizens", color="black")
        plt.scatter(detected[:, 0], detected[:, 1], label= "Prediction for "+ planet[curplan], color="red", marker='x', s=20)
        plt.legend()
        plt.xlabel('Weight (pounds)', loc='left')
        plt.ylabel('Height (inches)')
        f1sc = f1_score_(planet_filter(Ytest, curplan), ret[0][curplan], pos_label=1.)
        st = correct_ratio(ret[0][curplan], Ytest, curplan)
        plt.title("%s vs All f1 score : %.2f\nprecision %d/%d : %.2f %% with %d false positiv and %d missed" % (planet[curplan], f1sc, st[0], st[1], st[2], st[3], st[4]))
        plt.suptitle("Best model Logistical Regression lambda=0.1\n1 vs ALL visualisation on test dataset")
    plt.show()

    true_data = []
    for planx in range(4):
        data = []
        for idx in range (Xtest.shape[0]):
            if (Ytest[idx][0] == float(planx)):
                data.append(test[idx])
        true_data.append(np.array(data))

    pred_data = []
    for planx in range(4):
        data = []
        for idx in range(Xtest.shape[0]):
            if (ret[1][idx][0] == float(planx)):
                data.append(test[idx])
        pred_data.append(np.array(data))

    errors = []
    for planx in range(4):
        data = []
        for idx in range(Xtest.shape[0]):
            if (ret[1][idx][0] != Ytest[idx][0]):
                errors.append(test[idx])

    errors = np.array(errors)

    colors = {0: "pink", 1: "brown", 2 : "red", 3 : "black"}

    stat = correct_model(ret[1], Ytest)

    plt.subplot(1, 2, 1)
    plt.title("Citizen classification of the test dataset")
    for idx in range(4):
        plt.scatter(true_data[idx][:, 0], true_data[idx][:, 1], label=planet[idx], color=colors[idx])
    plt.legend()
    plt.xlabel('Weight (pounds)', loc='left')
    plt.ylabel('Height (inches)')
    plt.subplot(1, 2, 2)
    plt.title("Citizen classification from the logistic regression")
    for idx in range(4):
        plt.scatter(pred_data[idx][:, 0], pred_data[idx][:, 1], label=planet[idx], color=colors[idx])
    plt.scatter(errors[:, 0], errors[:, 1], label="errors", color="blue", marker='x', s=10)
    plt.legend()
    plt.xlabel('Weight (pounds)', loc='left')
    plt.ylabel('Height (inches)')
    plt.subplots_adjust(top=0.8)
    plt.suptitle(f"Logistic regression prediction and tested data comparaison\nVisualisation displaying weight in respect of the height\nOverall Precision : {stat[0]}/{stat[1]} {stat[2]} %")
    plt.show()

    print("Evaluating all model on test dataset :")

    for curplan in range(4):

        plt.subplot(2, 2, curplan + 1)
        plt.xlabel("lambdas", loc='left')
        plt.ylabel("f1 score")

        f1p = []
        for lamb in lambdas:
            print(f"f1 score of model lambda={lamb}, {planet[curplan]} vs ALL : {modelr[curplan][lamb][2]}")
            f1p.append(modelr[curplan][lamb][2])
        plt.bar(lambdas, f1p, width = 0.05, color='mediumspringgreen')
        plt.title(f"{planet[curplan]} vs ALL logistical regression\nMax: lambda={lambdas[f1p.index(max(f1p))]}")
    plt.suptitle("Displaying f1 score on the test dataset \nfor various lambdas of the logistical regression")
    plt.show()
    



