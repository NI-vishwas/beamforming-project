#!/usr/bin/env python3
#
#  beam_forming_weights.py
#
#  Copyright 2025 vishwas <vishwasks.reach@gmail.com>
#

import numpy as np
from utilities import load_dataset, get_project_directories, calculate_mrc_weights, combine_real_imag_weights, reshape_combined_weights

# Example Usage
dataset_dir, output_dir, data_tmp_dir = get_project_directories()
dataset, target_weights = load_dataset(data_tmp_dir+'/dataset.pkl', data_tmp_dir+'/target_weights.pkl')

ue_data = dataset[0]['user']['channel']
bs_data = dataset[0]['basestation']

mrc_weights = calculate_mrc_weights(ue_data, bs_data)
real_weights, imag_weights = get_real_imag_weights(mrc_weights)

print(f"MRC Weights shape: {mrc_weights.shape}")
print(f"Real Weights shape: {real_weights.shape}")
print(f"Imaginary Weights shape: {imag_weights.shape}")

# Access weights for a specific UE and antenna
ue_index = 0
antenna_index = 30
print(f"Real part of MRC weight for UE {ue_index}, antenna {antenna_index}: {real_weights[ue_index, antenna_index]}")
print(f"Imaginary part of MRC weight for UE {ue_index}, antenna {antenna_index}: {imag_weights[ue_index, antenna_index]}")

# Example usage (assuming you have real_weights and imag_weights from the previous code)
combined_weights = combine_real_imag_weights(real_weights, imag_weights)
reshaped_weights = reshape_combined_weights(combined_weights)

print(f"Combined Weights shape: {combined_weights.shape}")
print(f"Reshaped Weights shape: {reshaped_weights.shape}")

# Access an example combined weight
print(f"Example combined weight for UE 0: {reshaped_weights[0]}")
