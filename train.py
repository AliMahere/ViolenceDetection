import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import Bidirectional
import numpy as np
import tensorflow as tf
from os import path
import os
from dataset_utils import get_sequenc
X,y = get_sequenc('dataset/x_positive.npy','dataset/x_negative.npy')

model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences=True),input_shape=(5, 5336)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X, y, epochs=8000, batch_size=64, verbose=2)