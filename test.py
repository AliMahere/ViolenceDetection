import cv2
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Bidirectional
from dataset_utils import get_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
from keras import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Bidirectional
from dataset_utils import get_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import metrics
from keras.utils import plot_model
from keras.models import load_model
model = load_model('HockyFight-11 أغسطس, 2020.h5')
model.summary()

test_size = 0.10
X,y = get_sequence('dataset/x_positive.npy','dataset/x_negative.npy')
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size, random_state=40)
y_pred = model.predict(X_test)
print("y_pred shape",y_pred.shape)
print("y shape",y.shape)
matrix = metrics.confusion_matrix(y_test, y_pred)
print(matrix)