import numpy as np
import os

def merge_dataset(dataset_path, output_path ):
    data_set = None
    for npy_file in os.listdir(dataset_path):
        if npy_file.endswith(".npy")  and data_set != None:
            data_sample = np.load(npy_file)
            data_set = np.vstack(data_set, data_sample)
        elif npy_file.endswith(".npy") and data_set == None:
            data_sample = np.load(npy_file)
            data_set = data_sample
        else:
            continue
    np.save(output_path + '/' "dataset.npy", data_set)

def get_sequence(x_path_positive,x_path_negative):

    return X,y