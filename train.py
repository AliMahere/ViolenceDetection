import tensorflow as tf
import keras
import numpy as np
import tensorflow as tf
from os import path
import os
from dataset_utils import get_sequenc
X,y = get_sequenc('dataset/x_positive.npy','dataset/x_negative.npy')

model = Sequential()
