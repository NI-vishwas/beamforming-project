#!/usr/bin/env python3
#
#  testing_data.py
#  
#  Copyright 2025 vishwas <vishwasks.reach@gmail.com>
#    

import sys
import argparse
import numpy as np
import subprocess  # Import subprocess
from pprint import pprint
from utilities import (
    get_project_directories,
    load_dataset,
    prepare_data,
    train_and_evaluate_model,
    save_model_and_scalers,
    train_and_evaluate_svr,
    load_model_and_scalers,
    predict_with_svr,
    generate_random_users,
    write_testdatatofile,
)

# test_users = [{
        # 'paths': [{
            # 'num_paths': 10,
            # 'DoD_phi': np.array([-59.8567, -59.8567, -125.881, -125.881, -44.6496, -44.6496, -139.264, -139.264, -23.3263, -23.3263], dtype=np.float32),
            # 'DoD_theta': np.array([92.7304, 94.9064, 92.5714, 94.6216, 92.2194, 94.4322, 92.0786, 94.1517, 91.2573, 92.2623], dtype=np.float32),
            # 'DoA_phi': np.array([120.143, 120.143, 125.435, 125.435, 44.6496, 44.6496, 41.1832, 41.1832, 156.227, 156.227], dtype=np.float32),
            # 'DoA_theta': np.array([87.2696, 94.9064, 87.4286, 94.6216, 87.7806, 94.4322, 87.9214, 94.1517, 88.7427, 92.2623], dtype=np.float32),
            # 'phase': np.array([-70.6505, 140.981, -130.761, 139.037, -164.918, 119.471, 31.0889, 102.059, 66.6077, -177.672], dtype=np.float32),
            # 'ToA': np.array([2.80186e-07, 2.80897e-07, 2.97491e-07, 2.98161e-07, 3.44641e-07, 3.45416e-07, 3.67982e-07, 3.68708e-07, 6.08277e-07, 6.08605e-07], dtype=np.float32),
            # 'power': np.array([1.0261941e-10, 4.9102108e-11, 3.8556679e-11, 2.0615764e-11, 3.2040562e-12, 3.9409429e-13, 3.4229455e-13, 1.0856750e-13, 4.7522568e-15, 3.8203215e-15], dtype=np.float32)
        # }],
        # 'distance': [92.6780014038086],
        # 'pathloss': [109.61599731445312],
        # 'LoS': [1]
    # }]
    
test_users = [{
        "paths": [
            {
                "num_paths": 2,
                "DoD_phi": [
                    -125.05482729400879,
                    -83.67059461979291
                ],
                "DoD_theta": [
                    91.8041649832501,
                    92.73793103431807
                ],
                "DoA_phi": [
                    123.73041053239128,
                    84.23683019593116
                ],
                "DoA_theta": [
                    91.83230880422214,
                    87.80240604291656
                ],
                "phase": [
                    56.78393680780201,
                    -8.462997297682506
                ],
                "ToA": [
                    7.481186192209859e-07,
                    6.371748477539719e-07
                ],
                "power": [
                    1.778691738640678e-11,
                    9.756421208231111e-11
                ]
            }
        ],
        "LoS": [
            0
        ],
        "distance": [
            90.89067989841288
        ],
        "pathloss": [
            0.2942182319884509
        ]
    }

]

if __name__ == '__main__':
	dataset_dir, output_dir, data_tmp_dir = get_project_directories()
	dataset, target_weights = load_dataset(data_tmp_dir+'/dataset.pkl', data_tmp_dir+'/target_weights.pkl')

	loaded_model, loaded_x_scaler, loaded_y_scaler = load_model_and_scalers(
	model_filename=data_tmp_dir + "/nn_beamforming_model.pkl", x_scaler_filename=data_tmp_dir + "/nn_x_scaler.pkl",
	y_scaler_filename=data_tmp_dir + "/nn_y_scaler.pkl")
	test_input = prepare_data(test_users[0], target_weights, max_paths=10)[0]
	predicted_weights_scaled = loaded_model.predict(test_input)
	predicted_weights = loaded_y_scaler.inverse_transform(predicted_weights_scaled)
	print(f"Predicted Weights (real, imaginary): {predicted_weights[0]}")
