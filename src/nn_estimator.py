#!/usr/bin/env python3
#
#  nn_estimator.py
#
#  Copyright 2025 vishwas <vishwasks.reach@gmail.com>
#
#


import sys
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold

def tune_rnn_hyperparameters(X_train, y_real_train, y_imag_train, param_grid=None, cv=3):
    """
    Tunes the hyperparameters for two RNN models (real and imaginary parts) using GridSearchCV-like logic.

    Args:
        X_train (numpy.ndarray): Training feature matrix.
        y_real_train (numpy.ndarray): Training target values for the real part.
        y_imag_train (numpy.ndarray): Training target values for the imaginary part.
        param_grid (dict, optional): Dictionary specifying the hyperparameter search space.
                                     If None, a default search space is used.
        cv (int, optional): Number of cross-validation folds. Defaults to 3.

    Returns:
        tuple: (real_best_model, imag_best_model, best_real_params, best_imag_params)
               where best_real_params and best_imag_params are dictionaries of the best parameters.
    """

    if param_grid is None:
        param_grid = {
            'units': [64, 128, 256],
            'learning_rate': [0.001, 0.01],
            'epochs': [20, 50, 100],
            'batch_size': [16, 32],
        }

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    best_real_mse = float('inf')
    best_imag_mse = float('inf')
    real_best_model = None
    imag_best_model = None
    best_real_params = {}
    best_imag_params = {}

    for units in param_grid['units']:
        for learning_rate in param_grid['learning_rate']:
            for epochs in param_grid['epochs']:
                for batch_size in param_grid['batch_size']:
                    real_mses = []
                    imag_mses = []

                    for train_index, val_index in kf.split(X_train):
                        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                        y_real_train_fold, y_real_val_fold = y_real_train[train_index], y_real_train[val_index]
                        y_imag_train_fold, y_imag_val_fold = y_imag_train[train_index], y_imag_train[val_index]

                        real_model = keras.Sequential([
                            keras.layers.Dense(units, activation='relu', input_shape=(X_train.shape[1],)),
                            keras.layers.Dense(units // 2, activation='relu'),
                            keras.layers.Dense(1)
                        ])

                        imag_model = keras.Sequential([
                            keras.layers.Dense(units, activation='relu', input_shape=(X_train.shape[1],)),
                            keras.layers.Dense(units // 2, activation='relu'),
                            keras.layers.Dense(1)
                        ])

                        real_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
                        imag_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

                        real_model.fit(X_train_fold, y_real_train_fold, epochs=epochs, batch_size=batch_size, verbose=0)
                        imag_model.fit(X_train_fold, y_imag_train_fold, epochs=epochs, batch_size=batch_size, verbose=0)

                        real_mse = real_model.evaluate(X_val_fold, y_real_val_fold, verbose=0)
                        imag_mse = imag_model.evaluate(X_val_fold, y_imag_val_fold, verbose=0)

                        real_mses.append(real_mse)
                        imag_mses.append(imag_mse)

                    mean_real_mse = np.mean(real_mses)
                    mean_imag_mse = np.mean(imag_mses)

                    if mean_real_mse < best_real_mse:
                        best_real_mse = mean_real_mse
                        real_best_model = keras.models.clone_model(real_model)
                        real_best_model.set_weights(real_model.get_weights())
                        best_real_params = {'units': units, 'learning_rate': learning_rate, 'epochs': epochs, 'batch_size': batch_size}

                    if mean_imag_mse < best_imag_mse:
                        best_imag_mse = mean_imag_mse
                        imag_best_model = keras.models.clone_model(imag_model)
                        imag_best_model.set_weights(imag_model.get_weights())
                        best_imag_params = {'units': units, 'learning_rate': learning_rate, 'epochs': epochs, 'batch_size': batch_size}

    print("Best Real MSE:", best_real_mse)
    print("Best Imag MSE:", best_imag_mse)
    print("Best Real Parameters:", best_real_params)
    print("Best Imag Parameters:", best_imag_params)

    return real_best_model, imag_best_model, best_real_params, best_imag_params

if __name__ == '__main__':
    # Example Usage
    num_users = 100
    num_features = 10
    X_train = np.random.rand(num_users, num_features)
    y_real_train = np.random.rand(num_users)
    y_imag_train = np.random.rand(num_users)

    real_best_model, imag_best_model, best_real_params, best_imag_params = tune_rnn_hyperparameters(X_train, y_real_train, y_imag_train)
