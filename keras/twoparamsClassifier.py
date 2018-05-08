import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from sklearn import datasets

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X, y = datasets.make_classification(n_samples=200, n_features=2, n_informative=2,
                                    n_redundant=0, n_classes=2, n_clusters_per_class=1,
                                    n_repeated=0)
model = Sequential()
model.add(Dense(units=1,input_shape=(2,)))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="sgd")

print("training")
for step in range(501):
  cost = model.train_on_batch(X, y)
  if step % 50 == 0:
        print("After %d trainings, the cost: %f" % (step, cost))

# 测试过程
print('\nTesting ------------')
cost = model.evaluate(X, y, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

y_pred = model.predict(X)
y_pred = (y_pred*2).astype('int')  
plt.subplot(2,1,1).scatter(X[:,0], X[:,1], c=y_pred[:,0])
plt.subplot(2,1,2).scatter(X[:,0], X[:,1], c=y)
plt.show()
# model.fit(X, y, batch_size=50, epochs=10)
# model.evaluate(X, y, batch_size=40)