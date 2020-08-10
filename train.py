from keras import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Bidirectional
from dataset_utils import get_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

X,y = get_sequence('dataset/x_positive.npy','dataset/x_negative.npy')
print("X shape", X.shape)
print("y shape", y.shape)

test_size = 0.10
batch_size=64
validation_split = 0.10

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size, random_state=40)
print("X_train shape ",X_train.shape)
print("X_test shape ",X_test.shape)

model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True),input_shape=(5, 5336)))
model.add(Bidirectional(LSTM(32)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split=validation_split, epochs=1000, batch_size=batch_size, verbose=0)
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
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# save model and architecture to single file
date_time = datetime.now()
d = date_time.strftime("%d %B, %Y")
model.save('HockyFight-'+d+'.h5')
print("Saved model to disk")