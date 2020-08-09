import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from os import path
import os
from dataset_utils import get_sequenc
X,y = get_sequenc('dataset/x_positive.npy','dataset/x_negative.npy')

model = Sequential()
model.add(LSTM(20, input_shape=(5, 5336), return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train LSTM
for epoch in range(1000):
    # fit model for one epoch on this sequence
    model.fit(X, y, epochs=1, batch_size=1, verbose=2)