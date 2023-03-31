from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as mat
import sys
from space_avocado import MyRidge
from space_avocado import add_polynomial_features
from space_avocado import data_spliter
from space_avocado import load_data
from space_avocado import polytrain
from pickle import *

if __name__ == "__main__":

    path_data='space_avocado.csv'
    df = load_data(path_data)
    if df is None:
        print(f"Error loading data from {path_data}")
        sys.exit()

    X = df[['weight', 'prod_distance', 'time_delivery']].to_numpy()
    Y = df[['target']].to_numpy()
    (Xtrain, Xcross, Xtest, Ytrain, Ycross, Ytest, Rtrain) = data_spliter(X, Y, 0.5, normilize=True)
    print("Spliting data, training : %d x %d, coss validation : %d x %d, testing : %d x %d" % (Xtrain.shape[0], Xtrain.shape[1], Xcross.shape[0],  Xcross.shape[1], Xtest.shape[0], Xtest.shape[1]))

    f = open("models.pickle", "wb")
    models={}

    for order in range(1, 5):
        mod = {}
        for lamb in np.arange(0, 1.2, 0.2):
            print("fitting order %d lamb %.2f" % (order, lamb))
            ret = polytrain(Xtrain, Ytrain, Xcross, Ycross, order, float(lamb))
            mod.update({round(float(lamb), 1) : ret})
        models.update({order : mod})

    dump(models, f)
    f.close()
