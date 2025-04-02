#!/usr/bin/env python3
#
#  data_insights.py
#
#  Copyright 2025 vishwas <vishwasks.reach@gmail.com>
#

import sys
from utilities import load_dataset, get_project_directories, prepare_data
from nn_estimator import tune_rnn_hyperparameters
from svr_estimator import tune_svr_hyperparameters
import numpy as np

def get_property_nd_types(dataset):
    user_prop_type = {}
    basestation_prop_type = {}

    for prop in dataset[0].keys():
        user_prop_type[prop] = type(dataset[0][prop])

    for prop in dataset[0]['user'].keys():
        user_prop_type[prop] = type(dataset[0]['user'][prop])

    for prop in dataset[0]['user']['paths'][0].keys():
        user_prop_type[prop] = type(dataset[0]['user']['paths'][0][prop])

    for prop in dataset[0]['basestation'].keys():
        basestation_prop_type[prop] = type(dataset[0]['basestation'][prop])

    return user_prop_type, basestation_prop_type

if __name__ == '__main__':
    dataset_dir, output_dir, data_tmp_dir = get_project_directories()
    dataset, target_weights = load_dataset(data_tmp_dir+'/dataset.pkl', data_tmp_dir+'/target_weights.pkl')

    # Number of users : 4500
    print(f"Number of UEs: {len(dataset[0]['user']['paths'])}")

    # Properties of Each UE
    props = [prop for prop in list(dataset[0]['user'].keys())]
    props.extend([prop for prop in list(dataset[0]['user']['paths'][0].keys())])

    user_prop_type, basestation_prop_type = get_property_nd_types(dataset)
    print(f"Properties available for UE")
    print(f"Property | Type of Data")
    print(f"---------|-------------")

    for prop, typ in user_prop_type.items():
        print(prop,'|',str(typ).strip('<class').strip('>') )
    print()

    print(f"Properties available for Base Station")
    print(f"Property | Type of Data")
    print(f"---------|-------------")

    for prop, typ in basestation_prop_type.items():
        print(prop,'|',str(typ).strip('<class').strip('>') )
    print()

    print(f"Size of channel matrix at user: {dataset[0]['user']['channel'].shape}")
    print(f"Size of channel matrix at basestation: {np.array(dataset[0]['basestation']['channel']).shape}")

    min_distance = np.min(dataset[0]['user']['distance'])
    max_distance = np.max(dataset[0]['user']['distance'])
    print(f"Minimum distance: {min_distance}")
    print(f"Maximum distance: {max_distance}")

    keys = ['DoD_phi','DoD_theta','DoA_phi','DoA_theta', 'phase', 'ToA','power']
    data = {}
    for key in keys:
        temp = []
        for indx in range(len(dataset[0]['user']['paths'])):
            temp.extend(dataset[0]['user']['paths'][indx][key].tolist())

        data['min_'+key] = min(temp)
        data['max_'+key] = max(temp)
        data['avg_'+key] = sum(temp)/len(temp)

    for k,v in data.items():
        print(f"{k}: {v}")
        
    # Parameter Estimates for RNN
    # print(f"Parameter Estimates for RNN: ")
    # X_scaled, y_scaled, x_scaler, y_scaler = prepare_data(dataset[0]['user'], target_weights)
    # real_best_model, imag_best_model, best_real_params, best_imag_params = tune_rnn_hyperparameters(X_scaled, y_scaled[:,0], y_scaled[:,1])
    
    # # Parameter Estimates for SVR
    # print(f"Parameter Estimates for SVR: ")
    # svr_real_model, svr_imag_model = tune_svr_hyperparameters(X_scaled, y_scaled[:,0], y_scaled[:,1])
    
