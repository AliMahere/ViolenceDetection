from keras import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Bidirectional
from dataset_utils import get_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X,y = get_sequence('dataset/x_positive.npy','dataset/x_negative.npy')
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.10, random_state=40)
model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences=True),input_shape=(5, 5336)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split=0.10, epochs=1000, batch_size=64, verbose=0)
# list all data in history0
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()