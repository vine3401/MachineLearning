import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

x_train = np.random.random((1000, 20))
y_train = np_utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

x_test = np.random.random((100, 20))
y_test = np_utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(20,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

sgd = SGD(lr=0.01, momentum=0.9, decay=1e-06, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200, batch_size=128)
model.evaluate(x_test, y_test, batch_size=128)