import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 样本数据集，第一列为x，第二列为y，在x和y之间建立回归模型
data=[
    [0.067732,3.176513],[0.427810,3.816464],[0.995731,4.550095],[0.738336,4.256571],[0.981083,4.560815],
    [0.526171,3.929515],[0.378887,3.526170],[0.033859,3.156393],[0.132791,3.110301],[0.138306,3.149813],
    [0.247809,3.476346],[0.648270,4.119688],[0.731209,4.282233],[0.236833,3.486582],[0.969788,4.655492],
    [0.607492,3.965162],[0.358622,3.514900],[0.147846,3.125947],[0.637820,4.094115],[0.230372,3.476039],
    [0.070237,3.210610],[0.067154,3.190612],[0.925577,4.631504],[0.717733,4.295890],[0.015371,3.085028],
    [0.335070,3.448080],[0.040486,3.167440],[0.212575,3.364266],[0.617218,3.993482],[0.541196,3.891471]
]

dataMat = np.array(data)
x = dataMat[:, 0]
y = dataMat[:, 1]
print(x.shape)

model = Sequential()
model.add(Dense(input_dim=1, units=1))
model.compile(loss="mse", optimizer='sgd')

print("training")
for step in range(501):
  cost = model.train_on_batch(x, y)
  if step % 50 == 0:
    print("After %d trainings, the cost: %f" % (step, cost))

print("testing")
cost = model.evaluate(x, y, batch_size=40)
print("test cost:", cost)
W, b = model.layers[0].get_weights()
print("Weights=", W, "\nbiases=", b)
y_pred = model.predict(x)
plt.scatter(x, y)
plt.plot(x, y_pred)
plt.show()