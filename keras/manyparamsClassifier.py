import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.utils import np_utils
from sklearn import datasets

X, y = datasets.make_classification(n_samples=2000, n_features=2, n_informative=2, 
                                    n_redundant=0,n_repeated=0,n_classes=3,
                                    n_clusters_per_class=1)
n_class = 3
y = np_utils.to_categorical(y, n_class)

model = Sequential()
model.add(Dense(input_shape=(2, ), units=n_class))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])

model.fit(X, y, batch_size=50, epochs=50)
cost = model.evaluate(X, y, batch_size=40)
print("test cost:", cost)
