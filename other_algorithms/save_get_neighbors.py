import os
import h5py
import numpy as np


# Store neighbors (indices, coords and dist) into a hdf5 file
def save_neighbors(indices, coords, dists, file_name):
    with h5py.File(file_name, 'w') as f:
        dset1 = f.create_dataset('indices', data=indices)
        dset2 = f.create_dataset('coords', data=coords)
        dset3 = f.create_dataset('dists', data=dists)
        print("Neighbors stored at " + file_name)


# Load neighbors (indices, coords and dist) from a hdf5 file
def get_neighbors(file_name):
    if not os.path.exists(file_name):
        print("File " + file_name + " does not exist")
        return 0, 0, 0
    else:
        with h5py.File(file_name, 'r') as hdf5_file:
            print("Loading neighbors")
            return np.array(hdf5_file['indices']), np.array(hdf5_file['coords']), np.array(hdf5_file['dists'])

# file_name = "../other_algorithms/NearestNeighbors/15nn_Brute_Force_gaussian_euclidean.hdf5"
# indices, coords, dists = get_neighbors(file_name)