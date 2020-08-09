import tensorflow as tf
import keras
import numpy as np
import tensorflow as tf
from os import path
import os

from random import random
from numpy import array
from numpy import cumsum


def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	return X, y

x , y = get_sequence(100)
print(x,y)
# def merge_dataset(dataset_path, output_path ):
#     data_set = None
#     for npy_file in os.listdir(dataset_path):
#         if npy_file.endswith(".npy")  and data_set != None:
#             data_sample = np.load(npy_file)
#             data_set = np.vstack(data_set, data_sample)
#         elif npy_file.endswith(".npy") and data_set == None:
#             data_sample = np.load(npy_file)
#             data_set = data_sample
#         else:
#             continue
#     np.save(output_path + '/' "dataset.npy", data_set)
