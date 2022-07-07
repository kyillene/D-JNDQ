import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr, spearmanr, kendalltau
from scipy.optimize import fmin
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.optimize import leastsq

# Following functions are used to calculate PCC, SROCC, KROCC, RMSE between the predicted similarity scores
# and ground truth dissimilariy scores. Implementation is acquired from "Suiyi  Ling" and currently working on cpu.


def logistic(t, x):
    return 0.5 - (1 / (1 + np.exp(t * x)))


def fitfun(t, x):
    res = t[0] * (logistic(t[1], (x-t[2]))) + t[3] + t[4] * x
    return res


def errfun(t, x, y):
    return np.sum(np.power(y - fitfun(t, x),2))


def fitfun_4para(t, x):
    res = t[0] * (logistic(t[1], (x-t[2]))) + t[3]
    return res


def errfun_4para(t, x, y):
    return np.sum(np.power(y - fitfun(t, x),2))


def RMSE(y_actual, y_predicted):
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    return rmse


def coeff_fit(Obj,y) :
    temp = pearsonr(Obj, y)
    t = np.zeros(5)
    t[2] = np.mean(Obj)
    t[3] = np.mean(y)
    t[1] = 1/np.std(Obj)
    t[0] = abs(np.max(y) - np.min(y))
    t[4] = -1
    signslope = 1
    if temp[1]<=0:
        t[0] *= -1
        signslope *= -1
    v = [t, Obj, y]
    tt = fmin(errfun, t, args=(Obj, y))
    fit = fitfun(tt, Obj)
    cc = pearsonr(fit, y)[0]
    srocc = spearmanr(fit, y).correlation
    krocc = kendalltau(fit, y).correlation
    rmse = RMSE( np.absolute(y), np.absolute(fit) )
    return fit, cc, srocc, krocc, rmse
