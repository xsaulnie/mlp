from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as mat
import sys
from solar_system_census import MyLogisticRegression
from solar_system_census import add_polynomial_features
from solar_system_census import data_spliter
from solar_system_census import load_data
from solar_system_census import planet_filter
from solar_system_census import prediction_filter
from solar_system_census import f1_score_
from pickle import *


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

    f = open("models.pickle", "wb")
    models={}

    for curplan in range(4):

        mod = {}
        Ytrain1 = planet_filter(Ytrain, curplan)
        for lamb in lambdas:
            mlr = MyLogisticRegression(np.array([[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]),max_iter=100000, alpha=1e-2, penalty='l2', lambda_=lamb)
            print(f"{planets[curplan]} 's logistical regression alpha=1e-2, lambda={lamb}, max_iteration=100000 from null thetas, fitting data...")
            Xpoly = add_polynomial_features(Xtrain, 3)
            mlr.fit_(Xpoly, Ytrain1)
            Yhat = mlr.predict_(add_polynomial_features(Xcross, 3))
            Y_hat = prediction_filter(Yhat, 0.4)
            f1sc = f1_score_(planet_filter(Ycross, curplan), Y_hat, pos_label=1.)

            Y_test = prediction_filter(mlr.predict_(add_polynomial_features(Xtest, 3)), 0.5)
            f1testscore = f1_score_(planet_filter(Ytest, curplan), Y_test, pos_label=1.)



            print("theta obtened : [[%.2f], [%.2f], [%.2f], [%.2f]]" % (mlr.theta[0][0], mlr.theta[1][0], mlr.theta[2][0], mlr.theta[3][0]))
            print("with a loss of %.6f" % (mlr.loss_(planet_filter(Ycross, curplan), Yhat)))
            print("f1 score, planet %s, lambda %.1f : %f" % (planet[curplan], lamb, f1sc))
            mod.update({lamb : (mlr.theta, f1sc, f1testscore)})

        models.update({curplan : mod})

    dump(models, f)
    f.close()