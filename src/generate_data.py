#!/usr/bin/env python3
#
#  01_gen_data.py
#  
#  Copyright 2025 vishwas <vishwasks.reach@gmail.com>
#

import DeepMIMO
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utilities import write_object_to_file, get_project_directories, calculate_mrc_weights, get_real_imag_weights

plt.rcParams['figure.figsize'] = [12, 8] # Set default plot size
dataset_dir, output_dir, data_tmp_dir = get_project_directories()

# Load the default parameters
parameters = DeepMIMO.default_params()

# Print the Default parameters
print(f"Default Parameters {parameters}")

# Setting up the scenario
# Set scenario name
parameters['scenario'] = 'O1_28'

# Set the main folder containing extracted scenarios
parameters['dataset_folder'] = dataset_dir
parameters['num_paths'] = 10

# To load the first five scenes, set
parameters['dynamic_settings']['first_scene'] = 1
parameters['dynamic_settings']['last_scene'] = 5

# User rows 1-100
parameters['user_row_first'] = 1
parameters['user_row_last'] = 100

# To activate the half of the users in each selected row randomly, set
parameters['user_subsampling'] = 0.5

# To activate the half of the selected rows randomly, set
parameters['row_subsampling'] = 0.5

# Activate only the first basestation
parameters['active_BS'] = np.array([1]) 

parameters['OFDM']['bandwidth'] = 0.05 # 50 MHz
parameters['OFDM']['subcarriers'] = 512 # OFDM with 512 subcarriers
parameters['OFDM']['subcarriers_limit'] = 64 # Keep only first 64 subcarriers

parameters['ue_antenna']['shape'] = np.array([1, 1, 1]) # Single antenna
parameters['bs_antenna']['shape'] = np.array([1, 32, 1]) # ULA of 32 elements
parameters['bs_antenna']['rotation'] = np.array([0, 30, 90]) # ULA of 32 elements
parameters['ue_antenna']['rotation'] = np.array([[0, 30], [30, 60], [60, 90]]) # ULA of 32 elements
parameters['ue_antenna']['radiation_pattern'] = 'isotropic' 
parameters['bs_antenna']['radiation_pattern'] = 'halfwave-dipole' 

pprint(f"Scenario Parameters {parameters}")

# Generate data
dataset = DeepMIMO.generate_data(parameters)

print(type(dataset))
# Plotting information
# Number of basestations
number_of_bs = len(dataset)
# Keys of a basestation dictionary
bs_dict = dataset[0].keys()
# Keys of a channel
channel_keys = dataset[0]['user'].keys()
# Number of UEs
no_of_ues =len(dataset[0]['user']['channel'])
# Shape of the channel matrix
shape_of_channel_matrix = dataset[0]['user']['channel'].shape


## Visualization of a channel matrix

csi_fig = plt.figure()
# Visualize channel magnitude response
# First, select indices of a user and bs
ue_idx = 0
bs_idx = 0
# Import channel
channel = dataset[bs_idx]['user']['channel'][ue_idx]
# Take only the first antenna pair
plt.imshow(np.abs(np.squeeze(channel).T))
plt.title('Channel Magnitude Response')
plt.xlabel('TX Antennas')
plt.ylabel('Subcarriers')
csi_fig.savefig(output_dir+'/csi_info.png')

## Visualization of the UE positions and path-losses
loc_x = dataset[bs_idx]['user']['location'][:, 0]
loc_y = dataset[bs_idx]['user']['location'][:, 1]
loc_z = dataset[bs_idx]['user']['location'][:, 2]
pathloss = dataset[bs_idx]['user']['pathloss']

# Path properties of BS 0 - UE 0
pprint(dataset[0]['user']['paths'][0])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
im = ax.scatter(loc_x, loc_y, loc_z, c=pathloss)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')


bs_loc_x = dataset[bs_idx]['basestation']['location'][:, 0]
bs_loc_y = dataset[bs_idx]['basestation']['location'][:, 1]
bs_loc_z = dataset[bs_idx]['basestation']['location'][:, 2]
ax.scatter(bs_loc_x, bs_loc_y, bs_loc_z, c='r')
ttl = plt.title('UE and BS Positions')
fig.savefig(output_dir+'/scenario.png')


# Writing dataset to a file
write_object_to_file(dataset, data_tmp_dir+'/dataset.pkl')

# Generating target weights
print(f"Shape of channel Matrix: {shape_of_channel_matrix}")
ue_data = dataset[0]['user']['channel']
bs_data = dataset[0]['basestation']

mrc_weights = calculate_mrc_weights(ue_data, bs_data)
reshaped_weights = get_real_imag_weights(mrc_weights)


# Access an example combined weight

write_object_to_file(reshaped_weights, data_tmp_dir+'/target_weights.pkl')
