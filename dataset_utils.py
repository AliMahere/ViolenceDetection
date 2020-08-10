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

def get_sequence(x_path_positive, x_path_negative):
    x_positive = np.load(x_path_positive)
    x_negative = np.load(x_path_negative)
    y_positive = np.ones((x_positive.shape[0]))
    y_negative = np.zeros((x_negative.shape[0]))
    X = np.vstack((x_negative,x_positive))
    y = np.hstack((y_negative,y_positive))
    return X,y