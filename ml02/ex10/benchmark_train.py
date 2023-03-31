from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as mat
import sys
from space_avocado import MyLinearRegression
from space_avocado import polynomial_features
from space_avocado import data_spliter
from space_avocado import load_data
from pickle import *

def poly_reg(X, Y, Xtest, Ytest, info, order):
    mylr = MyLinearRegression(thetas=info["theta"], alpha=info["alpha"], max_iter=info["iter"])
    Xp = polynomial_features(X, order)
    ret = mylr.fit_(Xp, Y)
    Y_hat = mylr.predict_(polynomial_features(Xtest, order))
    mse=MyLinearRegression.mse_(Ytest, Y_hat)
    r2 = MyLinearRegression.r2score_(Ytest, Y_hat)
    return (mylr.thetas, mse, r2)

if __name__ == "__main__":

    
    path_data='space_avocado.csv'
    df = load_data(path_data)
    if df is None:
        print(f"Error loading data from {path_data}")
        sys.exit()
    
    X = df[['weight', 'prod_distance', 'time_delivery']].to_numpy()
    Y = df[['target']].to_numpy()

    (Xtrain, Xtest, Ytrain, Ytest, Rtrain, Rtest) = data_spliter(X, Y, 0.5, normilize=True)


    f = open ("models.pickle","wb")
    models = {}

    for it in range(4):
        print("Space Avocado's Price Polynomial Linear Regression of order %d, alpha = 0.01, iteration = 2000000, thetas starts at 0" % (it + 1))
        ret = poly_reg(Xtrain, Ytrain,  Xtest, Ytest, {"theta" : np.zeros(((it + 1) * 3 + 1, 1)), "alpha" : 1e-2, "iter" : 2000000}, it + 1) #2000000}
        models.update({it + 1 : ret})
    dump(models, f)
    f.close()

