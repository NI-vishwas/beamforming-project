#!/usr/bin/env python3
#
#  utilities.py
#
#  Copyright 2025 vishwas <vishwasks.reach@gmail.com>
#
import sys
import joblib
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import sklearn as sk #Import scikit-learn
from pathlib import Path
import json
from pprint import pprint
import keras
from keras.src.legacy.saving import legacy_h5_format
import random
from dataclasses import dataclass, field
from typing import List, Union


def get_project_directories():
    """
    Determines the project's output and temporary data directories as strings.

    Assumes the project root is two levels above the current file's directory.

    Returns:
        tuple: A tuple containing the output directory and temporary data directory as strings.
    """
    current_file_path = Path(__file__).resolve()
    project_dir = current_file_path.parent.parent

    output_dir = project_dir / 'output'
    data_tmp_dir = project_dir / 'tmp'
    dataset_dir = project_dir / 'dataset'

    return str(dataset_dir),str(output_dir), str(data_tmp_dir)

def load_dataset(dataset_file, target_weights_file):
    """
    Loads a dataset from a file.

    This function reads a dataset from the specified file using the `read_object_from_file` function.

    Args:
        dataset_file (str): The path to the dataset file.

    Returns:
        object: The loaded dataset object.

    Prints:
        "Loading Data" to the console.

    Example:
        >>> dataset = load_dataset("my_dataset.pkl")
        Loading Data
        >>> type(dataset) # doctest: +SKIP
        <class 'list'> # Or whatever type your dataset is.
    """
    print("Loading Data")
    dataset = read_object_from_file(dataset_file)
    target_weights = read_object_from_file(target_weights_file)
    return dataset, target_weights

def train_and_evaluate_svr(x_scaled, y_scaled):
    """
    Trains two separate SVR models for the real and imaginary parts of y_scaled,
    and returns the predictions.

    Args:
        X (numpy.ndarray): Feature matrix.
        y_scaled (numpy.ndarray): Scaled target matrix, shape (num_samples, 2).

    Returns:
        numpy.ndarray: Array of complex predictions, shape (num_test_samples,).
    """

    y_real = y_scaled[:, 0]  # Real parts
    y_imag = y_scaled[:, 1]  # Imaginary parts
    
    # print(f"X Scaled [2]:{x_scaled[:2]}")
    print(f"Y Scaled Shape:{y_real.shape}")
    print("y_scaled shape:", y_scaled.shape)
    
    X_train, X_test, y_real_train, y_real_test, y_imag_train, y_imag_test = train_test_split(
        x_scaled, y_real, y_imag, test_size=0.2, random_state=42
    )
    
    # {'sigmoid', 'linear', 'rbf', 'precomputed', 'poly'}
    svr_real = SVR(kernel='poly')
    svr_imag = SVR(kernel='poly')

    svr_real_op = svr_real.fit(X_train, y_real_train)
    svr_imag_op = svr_imag.fit(X_train, y_imag_train)

    return svr_real_op, svr_imag_op

def predict_with_svr(svr_real, svr_imag, x_scaler, y_scaler, input_data):
    """Predicts real and imaginary parts using trained SVR models."""
    input_scaled = x_scaler.transform(input_data)
    predicted_real_scaled = svr_real.predict(input_scaled)
    predicted_imag_scaled = svr_imag.predict(input_scaled)

    # Inverse transform
    predicted_real = y_scaler.inverse_transform(np.column_stack((predicted_real_scaled, np.zeros_like(predicted_real_scaled))))[:, 0]
    predicted_imag = y_scaler.inverse_transform(np.column_stack((np.zeros_like(predicted_imag_scaled), predicted_imag_scaled)))[:, 1]
    

    return predicted_real, predicted_imag

    
def calculate_mrc_weights(ue_data, bs_data):
    """
    Calculates MRC weights for each UE, retaining the 64 subcarrier dimension.
    """

    num_ues = len(ue_data)
    num_antennas = bs_data['channel'][0].shape[1] if bs_data['channel'] else 1 #get antenna number from bs data.

    print(f"no. ues: {num_ues}, no. antennas: {num_antennas}")

    mrc_weights = np.zeros((num_ues, num_antennas, 64), dtype=complex)

    for ue_idx, ue in enumerate(ue_data):
        if ue is not None and len(ue.shape) == 3:
            channel = ue[0]  # Remove the batch dimension (1).
            if channel.shape[0] == num_antennas:
                mrc_weights[ue_idx, :, :] = np.conjugate(channel)
            elif channel.shape[1] == num_antennas:
                mrc_weights[ue_idx, :, :] = np.conjugate(channel.T)
            else:
                print(f"Warning: Channel shape mismatch for UE {ue_idx}. Expected ({num_antennas}, 64) or (64, {num_antennas}), got {channel.shape}.")
                mrc_weights[ue_idx, :, :] = np.ones((num_antennas, 64), dtype=complex)
        else:
            print(f"Warning: Invalid channel shape for UE {ue_idx}.")
            mrc_weights[ue_idx, :, :] = np.ones((num_antennas, 64), dtype=complex)

    return mrc_weights


def get_real_imag_weights(mrc_weights):

    """
    Splits complex MRC weights into real and imaginary parts.

    Args:
        mrc_weights (numpy.ndarray): Array of MRC weights, shape (num_ues, num_antennas).

    Returns:
        tuple: Tuple containing real and imaginary parts of the weights, both as numpy.ndarrays.
    """

    real_weights = np.real(mrc_weights)
    imag_weights = np.imag(mrc_weights)

    return real_weights, imag_weights


def prepare_data(data, target_weights, max_paths=10):
    """Prepares the input data for the neural network with variable num_paths.
    Modified to handle target_weights with shape (num_ues, num_antennas, 64)
    """
    print("In Prepare Data")
    input_data = []
    target_weights_arr = []

    data_len = len(data['paths'])
    print(f"Number of Users: {data_len}")
    #pprint(f"Test user data: {data}")

    for indx in range(data_len):
        user = data['paths'][indx]
        num_paths = user['num_paths']
        dod_phi_list = user['DoD_phi'].tolist()
        padded_dod_phi = dod_phi_list + [0] * (max_paths - num_paths)

        dod_theta_list = user['DoD_theta'].tolist()
        padded_dod_theta = dod_theta_list + [0] * (max_paths - num_paths)

        doa_phi_list = user['DoA_phi'].tolist()
        padded_doa_phi = doa_phi_list + [0] * (max_paths - num_paths)

        doa_theta_list = user['DoA_theta'].tolist()
        padded_doa_theta = doa_theta_list + [0] * (max_paths - num_paths)

        phase_real_list = [p.real for p in user['phase'].tolist()]
        padded_phase_real = phase_real_list + [0] * (max_paths - num_paths)

        phase_imag_list = [p.imag for p in user['phase'].tolist()]
        padded_phase_imag = phase_imag_list + [0] * (max_paths - num_paths)

        toa_list = user['ToA'].tolist()
        padded_toa = toa_list + [0] * (max_paths - num_paths)

        power_list = user['power'].tolist()
        padded_power = power_list + [0] * (max_paths - num_paths)

        flattened_user_data = [
            num_paths,
            *padded_dod_phi,
            *padded_dod_theta,
            *padded_doa_phi,
            *padded_doa_theta,
            *padded_phase_real,
            *padded_phase_imag,
            *padded_toa,
            *padded_power,
            data['distance'][indx],
            data['pathloss'][indx],
            data['LoS'][indx]
        ]
        input_data.append(flattened_user_data)

    X = np.array(input_data)

    if target_weights:
        print("In Prepare Data.....")
        for i, weight in enumerate(target_weights):
            if hasattr(weight, 'shape'):
                print(f"target_weights[{i}] shape: {weight.shape}")
            else:
                print(f"target_weights[{i}] is not a numpy array.")

    # Flatten target_weights for each user
    flattend_arr0 = target_weights[0].flatten()
    flattend_arr1 = target_weights[1].flatten()
    
    # print(f"flattend_arr0 shape: {flattend_arr0.shape}")
    
    y = np.array([(flattend_arr0[i], flattend_arr1[i]) for i in range(data_len)])
    
    print(f"y shape: {y.shape}")


    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    print(f"y_scaled Shape: {y_scaled.shape}")

    return X_scaled, y_scaled, scaler_x, scaler_y

def create_rnn_network(input_shape, output_shape): #added output_shape as an input.
    """Creates an RNN model for beamforming weight regression."""
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_shape[0], 1), input_shape=input_shape),  # Reshape for RNN input
        tf.keras.layers.LSTM(64, activation='relu', return_sequences=True), #lstm layer
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32, activation='relu'), #lstm layer
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='linear')  # Output: flattened weights, changed to output_shape.
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_evaluate_nn(X_scaled, y_scaled, x_scaler, y_scaler, epochs=100, batch_size=64):
    """Trains and evaluates the RNN model using pre-scaled data."""
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    model = create_rnn_network((X_train.shape[1],), y_train.shape[1]) #added y_train.shape[1] to the function call.
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    return model, x_scaler, y_scaler

def write_object_to_file(obj, filename):
    """
    Writes a Python object to a file using pickling.

    Args:
        obj: The Python object to be written.
        filename: The name of the file to write to.  It's good practice to
                  include the file extension (e.g., "my_object.pkl").
    """
    try:
        with open(filename, 'wb') as f:  # 'wb' for write binary
            pickle.dump(obj, f)
        print(f"Object successfully written to {filename}")
    except Exception as e:
        print(f"Error writing object to file: {e}")


def read_object_from_file(filename):
    """
    Reads a Python object from a file using pickling.

    Args:
        filename: The name of the file to read from.

    Returns:
        The Python object read from the file, or None if an error occurs.
    """
    try:
        with open(filename, 'rb') as f:  # 'rb' for read binary
            obj = pickle.load(f)
        print(f"Object successfully read from {filename}")
        return obj
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None
    except Exception as e:
        print(f"Error reading object from file: {e}")
        return None


@keras.saving.register_keras_serializable()
def save_model_and_scalers(model, x_scaler, y_scaler, model_filename="beamforming_model.keras", x_scaler_filename="x_scaler.pkl", y_scaler_filename="y_scaler.pkl"):
    """
    Saves the trained TensorFlow Keras model (in .keras format) and scalers to pickle files.

    Args:
        model: The trained TensorFlow Keras model.
        x_scaler: The StandardScaler object for input features.
        y_scaler: The StandardScaler object for target weights.
        model_filename: Filename for the saved model (default: "beamforming_model.keras").
        x_scaler_filename: Filename for the x_scaler (default: "x_scaler.pkl").
        y_scaler_filename: Filename for the y_scaler (default: "y_scaler.pkl").
    """
    # Save the model (using .keras format)
    #model.save(model_filename)
    keras.saving.save_model(model, model_filename)
    print(f"Model saved to {model_filename}")

    # Save the x_scaler
    with open(x_scaler_filename, 'wb') as f:
        pickle.dump(x_scaler, f)
    print(f"x_scaler saved to {x_scaler_filename}")

    # Save the y_scaler
    with open(y_scaler_filename, 'wb') as f:
        pickle.dump(y_scaler, f)
    print(f"y_scaler saved to {y_scaler_filename}")

@keras.saving.register_keras_serializable()
def load_model_and_scalers(model_filename="beamforming_model.keras", x_scaler_filename="x_scaler.pkl", y_scaler_filename="y_scaler.pkl"):
    """Loads the model and scalers from files."""

    # Load the model (using .keras format)
    # loaded_model = tf.keras.models.load_model(model_filename)
    loaded_model = legacy_h5_format.load_model_from_hdf5(model_filename, custom_objects = {'mse':'mse'})
    print(f"Model loaded from {model_filename}")

    # Load the x_scaler
    with open(x_scaler_filename, 'rb') as f:
        loaded_x_scaler = pickle.load(f)
    print(f"x_scaler loaded from {x_scaler_filename}")

    # Load the y_scaler
    with open(y_scaler_filename, 'rb') as f:
        loaded_y_scaler = pickle.load(f)
    print(f"y_scaler loaded from {y_scaler_filename}")

    return loaded_model, loaded_x_scaler, loaded_y_scaler

def plot_2d_from_3d(coordinates, filename="location_plot.png"):
    """
    Plots latitude and longitude points on a 2D plane from 3D coordinates.

    Args:
        coordinates: A NumPy ndarray of shape (4500, 3) containing latitude, longitude, and altitude.
    """
    if coordinates.shape != (4500, 3):
        raise ValueError("Coordinates must have shape (4500, 3)")

    latitudes = coordinates[:, 0]
    longitudes = coordinates[:, 1]


    plt.figure(figsize=(10, 8))
    plt.scatter(longitudes, latitudes, s=10)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("2D Plot of Latitude and Longitude")
    plt.grid(True)
    plt.savefig(filename) #save the image
    plt.close() #close the plot to prevent it from displaying

def generate_test_data(dataset, target_weights, num_test_users=5):
    """Generates test user data and target weights from an existing dataset."""

    test_users = []
    basestations = []
    test_target_weights = []
    paths = []
    distances = []
    los = []
    pathloss = []
    channel = []
    
    user_data = {}
    if not dataset:
        raise ValueError("Dataset is empty.")

    #pprint(dataset)
    num_users_in_dataset = dataset['user']['paths'].shape[0]

    if num_test_users > num_users_in_dataset:
        raise ValueError("Number of test users exceeds the number of users in the dataset.")

    selected_indices = random.sample(range(num_users_in_dataset), num_test_users)
    pprint(selected_indices)

    # print(f"Type of dataset paths: {type(dataset['user']['paths'][selected_indices[0]])}")
    # print(dataset['user']['paths'][selected_indices[0]])
    for idx in selected_indices:
        paths.append(dataset['user']['paths'][idx])
        distances.append(dataset['user']['distance'][idx])
        los.append(dataset['user']['LoS'][idx])
        pathloss.append(dataset['user']['pathloss'][idx])
        channel.append(dataset['user']['channel'][idx])
        
    user_data["paths"] = np.array(paths)
    user_data["distance"] = np.array(distances)
    user_data['LoS'] = np.array(los)
    user_data['pathloss'] = np.array(pathloss)
    user_data['channel'] = np.array(channel)
    
    basestation_data = dataset['basestation']
    #pprint(f"Base Station data")
    #pprint(dataset['basestation']['paths'])
    basestation_data = dataset['basestation']

    test_users.append(user_data)
    #print(f"Type of User Data: {test_users[0]['channel'].shape}")


    # Assuming calculate_mrc_weights and get_real_imag_weights are defined elsewhere.
    mrc_weights = calculate_mrc_weights(test_users[0]['channel'], basestation_data)
    test_target_weights = get_real_imag_weights(mrc_weights)
    
    
    # print(f"Test Target Weights in generate data: {test_target_weights}")
    return test_users, test_target_weights

def write_testdatatofile(data, filename='data.json'):
    """
    Writes test data containing numpy arrays to a JSON file.

    Args:
        data (list): The data to be written to the file.
        filename (str): The name of the output JSON file. Defaults to 'data.json'.
    """

    def convert_numpy_float(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        raise TypeError(repr(obj) + " is not JSON serializable")

    # Convert numpy arrays to lists
    def convert_numpy_arrays(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def recursive_convert(obj):
      if isinstance(obj, dict):
        return {key: recursive_convert(value) for key, value in obj.items()}
      elif isinstance(obj, list):
        return [recursive_convert(item) for item in obj]
      else:
        return convert_numpy_arrays(obj)

    converted_data = recursive_convert(data)

    # Write the data to a JSON file
    with open(filename, 'w') as f:
        json.dump(converted_data, f, default=convert_numpy_float, indent=4)

    print(f"Data written to {filename}")

def read_testdatafromfile(filename='data.json'):
    """
    Reads test data from a JSON file and converts lists back to numpy arrays.

    Args:
        filename (str): The name of the input JSON file. Defaults to 'data.json'.

    Returns:
        list: The data read from the file, with lists converted to numpy arrays.
    """

    def convert_lists_to_arrays(obj):
        if isinstance(obj, list):
            try:
                return np.array(obj, dtype=np.float32)
            except ValueError:
                return obj #If it can't be converted to a numpy array, return the list.
        return obj

    def recursive_convert(obj):
      if isinstance(obj, dict):
        return {key: recursive_convert(value) for key, value in obj.items()}
      elif isinstance(obj, list):
        return [recursive_convert(item) for item in obj]
      else:
        return convert_lists_to_arrays(obj)

    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return recursive_convert(data)

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{filename}'.")
        return None
        
def save_svr_and_scalers(svr_real, svr_imag, x_scaler, y_scaler, real_model_filename="svr_real.joblib", imag_model_filename="svr_imag.joblib", x_scaler_filename="x_scaler.pkl", y_scaler_filename="y_scaler.pkl"):
    """
    Saves the trained scikit-learn SVR models and scalers to files.

    Args:
        svr_real: The trained SVR model for the real part.
        svr_imag: The trained SVR model for the imaginary part.
        x_scaler: The StandardScaler object for input features.
        y_scaler: The StandardScaler object for target weights.
        real_model_filename: Filename for the saved real part SVR model.
        imag_model_filename: Filename for the saved imaginary part SVR model.
        x_scaler_filename: Filename for the x_scaler.
        y_scaler_filename: Filename for the y_scaler.
    """
    # Save the SVR models
    joblib.dump(svr_real, real_model_filename)
    print(f"Real SVR model saved to {real_model_filename}")
    joblib.dump(svr_imag, imag_model_filename)
    print(f"Imaginary SVR model saved to {imag_model_filename}")

    # Save the scalers
    with open(x_scaler_filename, 'wb') as f:
        pickle.dump(x_scaler, f)
    print(f"x_scaler saved to {x_scaler_filename}")
    with open(y_scaler_filename, 'wb') as f:
        pickle.dump(y_scaler, f)
    print(f"y_scaler saved to {y_scaler_filename}")

def load_svr_and_scalers(real_model_filename="svr_real.joblib", imag_model_filename="svr_imag.joblib", x_scaler_filename="x_scaler.pkl", y_scaler_filename="y_scaler.pkl"):
    """
    Loads saved scikit-learn SVR models and scalers from files.

    Args:
        real_model_filename: Filename of the saved real part SVR model.
        imag_model_filename: Filename of the saved imaginary part SVR model.
        x_scaler_filename: Filename of the x_scaler.
        y_scaler_filename: Filename of the y_scaler.

    Returns:
        tuple: (loaded_svr_real, loaded_svr_imag, loaded_x_scaler, loaded_y_scaler)
    """
    # Load the SVR models
    loaded_svr_real = joblib.load(real_model_filename)
    print(f"Real SVR model loaded from {real_model_filename}")
    loaded_svr_imag = joblib.load(imag_model_filename)
    print(f"Imaginary SVR model loaded from {imag_model_filename}")

    # Load the scalers
    with open(x_scaler_filename, 'rb') as f:
        loaded_x_scaler = pickle.load(f)
    print(f"x_scaler loaded from {x_scaler_filename}")
    with open(y_scaler_filename, 'rb') as f:
        loaded_y_scaler = pickle.load(f)
    print(f"y_scaler loaded from {y_scaler_filename}")

    return loaded_svr_real, loaded_svr_imag, loaded_x_scaler, loaded_y_scaler

def plot_user_weights(test_target_real,test_target_imag, pred_real, pred_imag, filename="weights_comparison.png", num_users=5):
    """
    Plots the test target weights and predicted weights for each user,
    with users on the x-axis and weights on the y-axis.

    Args:
        test_target_weights (numpy.ndarray): Array of test target weights (num_users, 32, 64).
        predicted_weights (numpy.ndarray): Array of predicted weights (num_users, 32, 64).
        filename (str): Filename to save the plot.
    """

    # Flatten the weights for each user to get a single value per user.
    # We can use the mean of the weights to represent each user.
    # test_real_means = np.mean(test_target_weights.real, axis=(1, 2))
    # pred_real_means = np.mean(predicted_weights.real, axis=(1, 2))
    # test_imag_means = np.mean(test_target_weights.imag, axis=(1, 2))
    # pred_imag_means = np.mean(predicted_weights.imag, axis=(1, 2))

    plt.figure(figsize=(12, 6))

    # Plot Real Parts
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_users + 1,1), test_target_real, label="Test Target (Real)")
    plt.plot(range(1, num_users + 1,1), pred_real, label="Predicted (Real)")
    plt.xlabel("User")
    plt.ylabel("Mean Real Weight")
    plt.title("Real Weights Comparison")
    plt.legend()

    # Plot Imaginary Parts
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_users + 1,1), test_target_imag, label="Test Target (Imag)")
    plt.plot(range(1, num_users + 1,1), pred_imag, label="Predicted (Imag)")
    plt.xlabel("User")
    plt.ylabel("Mean Imaginary Weight")
    plt.title("Imaginary Weights Comparison")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
