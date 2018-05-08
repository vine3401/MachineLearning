# -*- encoding:utf-8 -*-

from keras.models import Sequential        ## 引入网络模型
from keras.layers import Dense, Activation ## 引入全连接层和激活器
from keras.optimizers import SGD           ## 引入优化器
# 网络模型搭建
# Sequential() 是一系列网络层按照顺序构成的栈

model = Sequential() 
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, trainable=False))
model.add(Activation('softmax'))
model.pop()

# =====================训练模型搭建=========================
# 完成模型的搭建后，我们需要使用.compile()方法来编译模型：
# loss损失函数，交叉熵。optimizer优化器，sgd随机梯度下降法进行网络训练，metrics评估模型，accuracy准确率作为评判结果
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 优化器
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)   # lr学习速率，momentum表示动量项，decay是学习速率的衰减系数(每个epoch衰减一次)，Nesterov的值是False或者True，表示使不使用Nesterov momentum
# 目标函数（损失函数）：mean_squared_error，mean_absolute_error，squared_hinge，hinge，binary_crossentropy对数损失函数，categorical_crossentropy多分类的对数损失函数
model.compile(loss='categorical_crossentropy', optimizer=sgd)


# ========================模型训练=============================
# 完成模型编译后，我们在训练数据上按batch进行一定次数的迭代来训练网络
model.fit(x_train, y_train, epochs=5, batch_size=32,shuffl=True)  #epochs迭代次数，batch_size每次迭代使用的样本数，shuffl训练集是否洗乱
# model.train_on_batch(x_batch, y_batch)   # 一次训练一个样本，可以用来处理超过机器内存的数据集

# 设置当损失函数不再下降时就停止训练
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)  # 提前结束训练的触发函数
hist = model.fit(x_train, y_train, validation_split=0.2, callbacks=[early_stopping])  # validation_split交叉验证的分割比， callbacks每次训练后的回调函数
print(hist.history) # 打印训练过程中损失函数的值以及其他度量指标
