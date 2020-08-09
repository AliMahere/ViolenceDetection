import numpy as np
import os

def merge_dataset(dataset_path, output_path , name = 'dataset'):
    data_set = None
    for npy_file in os.listdir(dataset_path):
        if npy_file.endswith(".npy")  and data_set is None :
            data_sample = np.load(dataset_path+npy_file)
            data_set = data_sample
        elif npy_file.endswith(".npy") :

            data_sample = np.load(dataset_path + npy_file)
            data_set = np.vstack((data_set, data_sample))
        else:
            continue
    np.save(output_path + '/' +name+".npy", data_set)

merge_dataset('output/nv/','dataset', name = "x_positive")
# def get_sequence(x_path_positive,x_path_negative):
#
#     return X,y