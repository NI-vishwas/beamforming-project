#!/usr/bin/env python3
#
#  deprecated.py
#  
#  Copyright 2025 vishwas <vishwasks.reach@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  


import sys



def generate_test_data(num_test_users = 5, target_weights_shape = (32,64)):
    test_users = []
    test_target_weights = []

    for _ in range(num_test_users):
        random_integer = np.random.randint(2, 10)
        user_data = {
            'paths': [{
                'num_paths': random_integer,
                'DoD_phi': np.array(np.random.uniform(-159.93800354003906, -21.965999603271484, random_integer).tolist()),
                'DoD_theta': np.array(np.random.uniform(90.88089752197266, 96.1365966796875, random_integer).tolist()),
                'DoA_phi': np.array(np.random.uniform(20.955799102783203, 157.1840057373047, random_integer).tolist()),
                'DoA_theta': np.array(np.random.uniform(86.85769653320312, 96.1365966796875, random_integer).tolist()),
                'phase': np.array((np.random.uniform(-179.99899291992188, 179.99099731445312, random_integer) + 1j * np.random.uniform(-179.99899291992188, 179.99099731445312, random_integer)).tolist()),
                'ToA': np.array(np.random.uniform(2.4348599936274695e-07, 8.681569738655526e-07, random_integer).tolist()),
                'power': np.array(np.random.uniform(5.3137241435354915e-22, 1.3593141889867155e-10, random_integer).tolist())
            }],
            'LoS': np.random.choice([0, 1], 1).tolist(),
            'distance': np.random.uniform(72.9718017578125, 101.89900207519531, 1).tolist(),
            'pathloss': np.random.rand(1).tolist()
        }
        test_users.append(user_data)
        #test_target_weights.append(np.random.rand(target_weights_shape[0],target_weights_shape[1]))
    
    mrc_weights = calculate_mrc_weights(user_data, user_data)
    test_target_weights = get_real_imag_weights(mrc_weights)

    return test_users, test_target_weights

def generate_test_data(dataset, target_weights, num_test_users=5):
    """Generates test user data and target weights from an existing dataset."""

    test_users = []
    basestations = []
    test_target_weights = []
    paths = []
    user_data = {}
    if not dataset:
        raise ValueError("Dataset is empty.")

    pprint(dataset)
    num_users_in_dataset = dataset['user']['paths'].shape[0]

    if num_test_users > num_users_in_dataset:
        raise ValueError("Number of test users exceeds the number of users in the dataset.")

    selected_indices = random.sample(range(num_users_in_dataset), num_test_users)

    for index in selected_indices:

        path_data = {
        "num_paths":dataset['user']['paths'][index]['num_paths'],
        "DoD_phi":dataset['user']['paths'][index]['DoD_phi'],
        "DoD_theta":dataset['user']['paths'][index]['DoD_theta'],
        "DoA_phi":dataset['user']['paths'][index]['DoA_phi'],
        "DoA_theta":dataset['user']['paths'][index]['DoA_theta'],
        "phase":dataset['user']['paths'][index]['phase'],
        "ToA":dataset['user']['paths'][index]['ToA'],
        "power":dataset['user']['paths'][index]['power']
        }
        
        paths.append(path_data)

        user_data["LoS"]=dataset['user']['LoS'][index],
        user_data["distance"]=dataset['user']['distance'][index],
        user_data["pathloss"]=dataset['user']['pathloss'][index]
        
        user_data["paths"] = paths
        test_users.append(np.array(user_data))
        
    basestation_data = dataset['basestation']
    pprint(f"Base Station data")
    pprint(dataset['basestation']['paths'])
    basestation_data = dataset['basestation']

    # Assuming calculate_mrc_weights and get_real_imag_weights are defined elsewhere.
    mrc_weights = calculate_mrc_weights(test_users, basestation_data) #dataset[index]['bs'] is assumed to be the bs data.
    real, imag = get_real_imag_weights(mrc_weights)
    test_target_weights.append((real, imag)) #flatten the real and imaginary weights.
    return test_users, test_target_weights


def main(args):
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
