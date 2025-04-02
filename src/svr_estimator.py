#!/usr/bin/env python3
#
#  svr_estimator.py
#  
#  Copyright 2025 vishwas <vishwasks.reach@gmail.com>
#  
#  


import sys
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

def tune_svr_hyperparameters(X_train, y_real_train, y_imag_train, param_grid=None, cv=5):
    """
    Tunes the hyperparameters for two SVR models (real and imaginary parts) using GridSearchCV.

    Args:
        X_train (numpy.ndarray): Training feature matrix.
        y_real_train (numpy.ndarray): Training target values for the real part.
        y_imag_train (numpy.ndarray): Training target values for the imaginary part.
        param_grid (dict, optional): Dictionary specifying the hyperparameter search space.
                                     If None, a default search space is used.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.

    Returns:
        tuple: (svr_real_model, svr_imag_model), where each is a trained SVR model
               with the best hyperparameters found by GridSearchCV.
    """

    if param_grid is None:
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 1, 10]
        }

    grid_search_real = GridSearchCV(SVR(), param_grid, cv=cv)
    grid_search_imag = GridSearchCV(SVR(), param_grid, cv=cv)

    grid_search_real.fit(X_train, y_real_train)
    grid_search_imag.fit(X_train, y_imag_train)

    print("Best Real Params:", grid_search_real.best_params_)
    print("Best Imag Params:", grid_search_imag.best_params_)

    svr_real_model = grid_search_real.best_estimator_
    svr_imag_model = grid_search_imag.best_estimator_

    return svr_real_model, svr_imag_model

if __name__ == '__main__':
    num_users = 100
    num_features = 10
    X_train = np.random.rand(num_users, num_features)
    y_real_train = np.random.rand(num_users)
    y_imag_train = np.random.rand(num_users)

    svr_real_model, svr_imag_model = tune_svr_hyperparameters(X_train, y_real_train, y_imag_train)

