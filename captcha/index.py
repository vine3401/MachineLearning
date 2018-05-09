import random
import string

import numpy as np
from keras.models import *
from keras.layers import *
import matplotlib.pyplot as plt
from captcha.image import ImageCaptcha


characters = string.ascii_uppercase + string.digits

width, height, n_len, n_class = 170, 80, 4, len(characters)
generator = ImageCaptcha(width=width, height=height)
random_str = ''.join([random.choice(characters) for j in range(4)])
img = generator.generate_image(random_str)

plt.imshow(img)
plt.title(random_str)

def gen(batch_size=32):
  X = np.zeros((batch_size, height, width, 3),dtype=np.uint8)
  y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
  generator = ImageCaptcha(width=width, height=height)
  while True:
    for i in range(batch_size):
      random_str = ''.join([random.choice(characters) for j in range(4)])
      X[i] = generator.generate_image(random_str)
      for j, ch in enumerate(random_str):
        y[j][i, :] = 0
        y[j][i, characters.find(ch)] = 1
    yield X, y

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])


input_tensor = Input((height, width, 3))
x = input_tensor
for i in range(4):
  x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
  x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
  x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
# 在训练过程中每次更新参数时随机断开一定百分比（rate）的输入神经元，防止过拟合。
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name="c%d" % (i+1))(x) for i in range(4)]
model = Model(input=input_tensor, output=x)
model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=['accuracy'])

model.fit_generator(gen(), samples_per_epoch=1600, nb_epoch=5,
                    nb_worker=2, pickle_safe=True, 
                    validation_data=gen(), nb_val_samples=1280)
X, y = next(gen(1))
y_pred = model.predict(X)
plt.title('real: %s\npred:%s'%(decode(y), decode(y_pred)))
plt.imshow(X[0], cmap='gray')
plt.show()


