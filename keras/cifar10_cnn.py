import os

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Activation, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import rmsprop

num_classes = 10
data_augmentation = True
batch_size = 32
epochs = 100

save_dir = os.path.join(os.getcwd(), "saved_models")
model_name = "keras_cifar10_cnn_trained_model.h5"


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
y_test /= 255

model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation("softmax"))


model.compile(loss=categorical_crossentropy,
              optimizer=rmsprop(lr=0.0001, decay=1e-6), metrics=["accuracy"])


if not data_augmentation:
    print("Not using data augmentation")
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, validation_data=(x_test, y_test),
              shuffle=True)
else:
    print("using data augmentation")
    datagen = ImageDataGenerator(featurewise_center=False, 
                                 samplewise_center=False, 
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 zca_whitening=False,
                                 rotation_range=0,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True,
                                 vertical_flip=False)
    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                     epochs=epochs, validation_data=(x_test, y_test),
                                     workers=4)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print("Saved trained model at %s" % model_path)


score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

