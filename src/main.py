#!/usr/bin/env python3
#
#  train_nd_test.py
#
#  Copyright 2025 vishwas <vishwasks.reach@gmail.com>
#

import sys
import argparse
import numpy as np
import subprocess  # Import subprocess
from pprint import pprint
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

from utilities import (
    get_project_directories,
    load_dataset,
    prepare_data,
    train_and_evaluate_nn,
    train_test_split,
    save_model_and_scalers,
    train_and_evaluate_svr,
    save_svr_and_scalers,
    load_model_and_scalers,
    load_svr_and_scalers,
    predict_with_svr,
    generate_test_data,
    write_testdatatofile,
    calculate_mrc_weights,
    get_real_imag_weights,
    plot_user_weights
)

NUM_TEST_USERS = 5
dataset_dir, output_dir, data_tmp_dir = get_project_directories()
dataset, target_weights = load_dataset(data_tmp_dir+'/dataset.pkl', data_tmp_dir+'/target_weights.pkl')
pprint(f"Generating data for {NUM_TEST_USERS} TEST USERS")
test_users, test_target_weights = generate_test_data(dataset[0], target_weights, NUM_TEST_USERS)
# pprint(test_users)
data_save_file = data_tmp_dir +"/data.json"
#write_testdatatofile(test_users,filename=data_save_file)
pprint(f"Writing Test Users data to file: {data_save_file}")

def train_save_nn():
    X_scaled, y_scaled, x_scaler, y_scaler = prepare_data(dataset[0]['user'], target_weights)
    model, x_scaler, y_scaler = train_and_evaluate_nn(X_scaled, y_scaled, x_scaler, y_scaler)
    save_model_and_scalers(model, x_scaler, y_scaler, model_filename=data_tmp_dir + "/nn_beamforming_model.h5", x_scaler_filename=data_tmp_dir + "/nn_x_scaler.pkl", y_scaler_filename=data_tmp_dir + "/nn_y_scaler.pkl")

def train_save_svr():
    X_scaled, y_scaled, x_scaler, y_scaler = prepare_data(dataset[0]['user'], target_weights)
    svr_real, svr_imag = train_and_evaluate_svr(X_scaled, y_scaled)
    # print(f"svr real: {svr_real}")
    save_svr_and_scalers(svr_real, svr_imag, x_scaler, y_scaler, real_model_filename=data_tmp_dir + "/svr_real.joblib", imag_model_filename=data_tmp_dir + "/svr_imag.joblib", x_scaler_filename=data_tmp_dir + "/svr_x_scaler.pkl", y_scaler_filename=data_tmp_dir + "/svr_y_scaler.pkl")

def test_nn():
    loaded_model, loaded_x_scaler, loaded_y_scaler = load_model_and_scalers(
        model_filename=data_tmp_dir + "/nn_beamforming_model.h5",
        x_scaler_filename=data_tmp_dir + "/nn_x_scaler.pkl",
        y_scaler_filename=data_tmp_dir + "/nn_y_scaler.pkl"
    )
    print(f"Type of test_users: {type(test_users)}")
    print(f"Number of test_users: {len(test_users[0]['paths'])}")
    X_scaled, y_scaled, x_scaler, y_scaler = prepare_data(test_users[0], test_target_weights)
    
    test_input = prepare_data(test_users[0], test_target_weights)[0]
    
    predicted_weights_scaled = loaded_model.predict(test_input)
    predicted_weights = loaded_y_scaler.inverse_transform(predicted_weights_scaled)
    
    pred_real_weights = predicted_weights[:, 0]
    pred_imag_weights = predicted_weights[:, 1]
    
    nn_real_mse = mean_squared_error(y_scaled[:,0], pred_real_weights)
    nn_imag_mse = mean_squared_error(y_scaled[:,1], pred_imag_weights)
    
    print(f"Predicted real NN Weights: {pred_real_weights}")
    print(f"Predicted imag NN Weights: {pred_imag_weights}")
    print(f"NN Real MSE: {nn_real_mse}")
    print(f"NN Imag MSE: {nn_imag_mse}")
    # print(f"Predicted imag NN Weights: {predicted_weights}")

def test_svr():
    """
    Loads trained SVR models and predicts real and imaginary weights for each test user.
    """
    loaded_svr_real, loaded_svr_imag, loaded_x_scaler, loaded_y_scaler = load_svr_and_scalers(
        real_model_filename=data_tmp_dir + "/svr_real.joblib",
        imag_model_filename=data_tmp_dir + "/svr_imag.joblib",
        x_scaler_filename=data_tmp_dir + "/svr_x_scaler.pkl",
        y_scaler_filename=data_tmp_dir + "/svr_y_scaler.pkl"
    )
    
    X_scaled, y_scaled, x_scaler, y_scaler = prepare_data(test_users[0], test_target_weights)
    
    test_input = prepare_data(test_users[0], test_target_weights, max_paths=10)[0] #assuming prepare_data returns a tuple, and that the first element of that tuple is the test input.

    NUM_TEST_USERS = test_input.shape[0] # dynamically get the number of test users.

    predicted_real_weights = []
    predicted_imag_weights = []

    for indx in range(NUM_TEST_USERS):
        predicted_real, predicted_imag = predict_with_svr(
            loaded_svr_real, loaded_svr_imag, loaded_x_scaler, loaded_y_scaler, test_input[indx].reshape(1,-1) #reshape to be 2d array, even if there is one test user.
        )
        predicted_real_weights.append(predicted_real[0])
        predicted_imag_weights.append(predicted_imag[0])

    # Convert lists to NumPy arrays
    predicted_real_weights = np.array(predicted_real_weights)
    predicted_imag_weights = np.array(predicted_imag_weights)
    
    svr_real_mse = mean_squared_error(y_scaled[:,0], predicted_real_weights)
    svr_imag_mse = mean_squared_error(y_scaled[:,1], predicted_imag_weights)

    # print(f"Predicted Real SVR Weights shape: {predicted_real_weights.shape}")
    # print(f"Predicted Imaginary SVR Weights shape: {predicted_imag_weights.shape}")
    print(f"Predicted Real SVR Weights: {predicted_real_weights}")
    print(f"Predicted Imaginary SVR Weights: {predicted_imag_weights}")
    print(f"SVR Real MSE: {svr_real_mse}")
    print(f"SVR Imag MSE: {svr_imag_mse}")
    
    
def evaluate_and_compare_models():
    """Evaluates and compares SVR and NN models, saves results, and plots."""
    X_scaled, y_scaled, x_scaler, y_scaler = prepare_data(test_users[0], test_target_weights)
    # Preparing the data
    test_input = prepare_data(test_users[0], test_target_weights, max_paths=10)[0] #assuming prepare_data returns a tuple, and that the first element of that tuple is the test input.
    NUM_TEST_USERS = test_input.shape[0] # dynamically get the number of test users.

    # SVR loading and Prediction
    loaded_svr_real, loaded_svr_imag, svr_x_scaler, svr_y_scaler = load_svr_and_scalers(
        real_model_filename=data_tmp_dir + "/svr_real.joblib",
        imag_model_filename=data_tmp_dir + "/svr_imag.joblib",
        x_scaler_filename=data_tmp_dir + "/svr_x_scaler.pkl",
        y_scaler_filename=data_tmp_dir + "/svr_y_scaler.pkl"
    )
    svr_real_pred = []
    svr_imag_pred = []

    for indx in range(NUM_TEST_USERS):
        predicted_real, predicted_imag = predict_with_svr(
            loaded_svr_real, loaded_svr_imag, svr_x_scaler, svr_y_scaler, test_input[indx].reshape(1,-1) #reshape to be 2d array, even if there is one test user.
        )
        svr_real_pred.append(predicted_real[0])
        svr_imag_pred.append(predicted_imag[0])

    # Convert lists to NumPy arrays
    svr_real_pred = np.array(svr_real_pred)
    svr_imag_pred = np.array(svr_imag_pred)

    # NN loading and Prediction
    nn_real_model, nn_x_scaler, nn_y_scaler = load_model_and_scalers(
        model_filename=data_tmp_dir + "/nn_beamforming_model.h5",
        x_scaler_filename=data_tmp_dir + "/nn_x_scaler.pkl",
        y_scaler_filename=data_tmp_dir + "/nn_y_scaler.pkl"
    )
    
    predicted_weights_scaled = nn_real_model.predict(test_input)
    predicted_weights = nn_y_scaler.inverse_transform(predicted_weights_scaled)
    
    nn_real_pred = predicted_weights[:, 0]
    nn_imag_pred = predicted_weights[:, 1]

    # Performance Evaluation
    svr_real_mse = mean_squared_error(y_scaled[:,0], svr_real_pred)
    svr_imag_mse = mean_squared_error(y_scaled[:,1], svr_imag_pred)
    nn_real_mse = mean_squared_error(y_scaled[:,0], nn_real_pred)
    nn_imag_mse = mean_squared_error(y_scaled[:,1], nn_imag_pred)

    # Save to CSV
    results_df = pd.DataFrame({
        'Model': ['SVR Real', 'SVR Imag', 'NN Real', 'NN Imag'],
        'MSE': [svr_real_mse, svr_imag_mse, nn_real_mse, nn_imag_mse]
    })
    results_df.to_csv(output_dir+'/model_comparison.csv', index=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Model'], results_df['MSE'])
    plt.ylabel('Mean Squared Error')
    plt.title('Model Comparison (MSE)')
    plt.savefig(output_dir+'/model_comparison_plot.png')
    
    plot_user_weights(y_scaled[:,0], nn_real_pred, y_scaled[:,1], nn_imag_pred, output_dir+"/nn_weights_comparison.png",5)
    plot_user_weights(y_scaled[:,0], svr_real_pred, y_scaled[:,1], svr_imag_pred, output_dir+"/svr_weights_comparison.png",5)
    
    print(f"Plots saved in {output_dir}")

    # Example comparison of first 5 predicted values.
    print(f"SVR Real Pred 5: {svr_real_pred[:5]}")
    print(f"NN Real Pred 5: {nn_real_pred[:5]}")
    print(f"SVR Imag Pred 5: {svr_imag_pred[:5]}")
    print(f"NN Imag Pred 5: {nn_imag_pred[:5]}")

def main():
    parser = argparse.ArgumentParser(description="Train or test a model.")

    parser.add_argument("--action", "-a", choices=["train", "test"], help="Specify the action: train or test.")
    parser.add_argument("--model", "-m", choices=["svr", "nn"], help="Specify the model: svr or nn.")
    parser.add_argument("--gen", "-g", action="store_true", help="Generate data independently.")
    parser.add_argument("--both", "-b", action="store_true", help="Test Compare Plots")

    args = parser.parse_args()

    if args.gen:
        print("Generating data...")
        try:
            subprocess.run(["python", "generate_data.py"], check=True) #execute generate_data.py
            print("Data generation completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during data generation: {e}")
        except FileNotFoundError:
            print("Error: generate_data.py not found.")
            
    elif args.both:
        print(f"Testing and Comparing models...")
        evaluate_and_compare_models()
    
    elif args.action and args.model:
        if args.action == "train" and args.model == "nn":
            print(f"Training {args.model} model...")
            # Your training logic here
            train_save_nn()
        elif args.action == "train" and args.model == "svr":
            print(f"Training {args.model} model...")
            # Your training logic here
            train_save_svr()
        elif args.action == "test" and args.model == "nn":
            print(f"Testing {args.model} model...")
            # Your testing logic here
            test_nn()
        elif args.action == "test" and args.model == "svr":
            print(f"Testing {args.model} model...")
            # Your testing logic here
            test_svr()
        
    elif args.action and not args.model :
        print("Error: Model type must be specified with --model or -m when using --action or -a.")
    elif args.model and not args.action :
        print("Error: Action must be specified with --action or -a when using --model or -m.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
