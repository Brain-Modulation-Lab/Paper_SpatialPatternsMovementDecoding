#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For a given regression model, search the optimal hyperameters and return the 
model

Created on Dec 2020
@author: Victoria Peterson
"""
#%%
from pyglmnet import GLM
from sklearn.linear_model import TweedieRegressor
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.pipeline import make_pipeline
#%%
def enet_train(alpha,l1_ratio,x,y):
    reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000,normalize=False)
    scaler = StandardScaler()
    clf = make_pipeline(scaler, reg)
    cval = cross_val_score(clf, x, y, scoring='r2', cv=3)
    cval[np.where(cval < 0)[0]] = 0
    return cval.mean()


def optimize_enet(x,y):
    """Apply Bayesian Optimization to select enet parameters."""
    def function(alpha, l1_ratio):
          
        return enet_train(alpha=alpha, l1_ratio=l1_ratio, x=x, y=y)
    
    optimizer = BayesianOptimization(
        f=function,
        pbounds={"alpha": (1e-6, 0.99), "l1_ratio": (1e-6,0.99)},
        random_state=0,
        verbose=0,
    )
    optimizer.probe(params=[1e-3, 1e-3], lazy=True)
    optimizer.maximize(n_iter=25, init_points=20, acq="ei", xi=1e-1)    
    return optimizer.max

def tweedie_train(alpha, power, x, y):
    reg = TweedieRegressor(alpha=alpha, power=power, max_iter=10000)
    scaler = StandardScaler()
    clf = make_pipeline(scaler, reg)
    cval = cross_val_score(clf, x, y, scoring='r2', cv=3)
    # cval[np.where(cval < 0)[0]] = 0
    return cval.mean()


def optimize_tweedie(x, y):
    """Apply Bayesian Optimization to select enet parameters."""
    def function(alpha, power):
        return tweedie_train(alpha=alpha, power=power, x=x, y=y)

    optimizer = BayesianOptimization(f=function,
                                      pbounds={"alpha": (0, 1), "power": (1, 1.99)},
                                      random_state=0,
                                      verbose=1)
    # optimizer.probe(params=[0.01, 1], lazy=True)
    optimizer.maximize(n_iter=25, init_points=20, acq="ei", xi=1e-1)
    return optimizer.max
def glm_train(alpha, reg_lambda, x, y):
    reg = GLM(distr="poisson", alpha=alpha, reg_lambda=reg_lambda,
              max_iter=10000, score_metric="pseudo_R2", tol=1e-4)
    scaler = StandardScaler()
    clf = make_pipeline(scaler, reg)
    cval = cross_val_score(clf, x, y, scoring='r2', cv=3)
    return cval.mean()


def optimize_glm(x, y):
    """Apply Bayesian Optimization to select enet parameters."""
    def function(alpha, reg_lambda):
        return glm_train(alpha=alpha, reg_lambda=reg_lambda, x=x, y=y)

    optimizer = BayesianOptimization(
        f=function,
        pbounds={"alpha": (1e-6, 1), "reg_lambda": (1e-6, 1)},
        random_state=0,
        verbose=0,
    )
    optimizer.maximize(n_iter=25, init_points=20, alpha=1e-3)
    return optimizer.max

def glm05_train(reg_lambda, x, y):
    reg = GLM(distr="poisson", alpha=0.5, reg_lambda=reg_lambda,
              max_iter=10000, score_metric="pseudo_R2", tol=1e-4)
    scaler = StandardScaler()
    clf = make_pipeline(scaler, reg)
    cval = cross_val_score(clf, x, y, scoring='r2', cv=3)
    return cval.mean()


def optimize_glm05(x, y):
    """Apply Bayesian Optimization to select enet parameters."""
    def function(reg_lambda):
        return glm05_train(reg_lambda=reg_lambda, x=x, y=y)

    optimizer = BayesianOptimization(
        f=function,
        pbounds={"reg_lambda": (1e-6,1)},
        random_state=0,
        verbose=0,
    )
    optimizer.maximize(n_iter=25, init_points=20, alpha=1e-3)
    return optimizer.max


def get_model(used_model, x, y):
    """get model.
    If used_model == 0: Enet
    If used_mode  == 1: Tweedie Regressor
    If used_model == 2: GLM with poisson
    if used_model == 3: GML with poisson, alpha 0.5
    

    """
    if used_model == 0:
        optimizer = optimize_enet(x, y)
        clf = ElasticNet(alpha=optimizer['params']['alpha'],
                         l1_ratio=optimizer['params']['l1_ratio'],
                         max_iter=10000)
        return clf, optimizer
    elif used_model == 1 or used_model == 'TWEEDIE':
        optimizer = optimize_tweedie(x, y)
        clf=TweedieRegressor(alpha=optimizer['params']['alpha'],
                             power=optimizer['params']['power'], max_iter=1000)
        return clf, optimizer
    elif used_model == 2 or used_model == 'GLM':
        optimizer = optimize_glm(x, y)
        clf = GLM(distr="poisson", alpha=optimizer['params']['alpha'],
                  reg_lambda=optimizer['params']['reg_lambda'], max_iter=10000)
        return clf, optimizer
    elif used_model == 3 or used_model == 'GLML05':
        optimizer = optimize_glm05(x, y)
        clf = GLM(distr="poisson", alpha=0.5,
                  reg_lambda=optimizer['params']['reg_lambda'], max_iter=10000)
        return clf, optimizer