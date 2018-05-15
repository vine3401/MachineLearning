import os

import numpy as np
from PIL import Image
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import rmsprop


dateset_path = "../datasets/SignLanguage"

def format_data(dataset={}):
    images = []
    labels = []
    num_classes = len(dataset.keys())
    for k, v in dataset.items():
        images = images + v
        labels = labels + list(np.ones(len(v)) * int(k))
    return (np.array(images), np.array(labels), num_classes)


def load_data(path=dateset_path):
    folders = os.listdir(path)
    folders_dict = {}
    for folder in folders:
        num_path = os.path.join(path, folder)
        images_path = [os.path.join(num_path, image) 
                        for image in os.listdir(num_path) 
                            if os.path.isfile(os.path.join(num_path, image))]
        data_item = []
        for image_path in images_path:
            image = Image.open(image_path)
            if image.size != (100, 100) or image.mode != "RGB":
                print(image_path)
                continue
            data_item.append(list(np.array(image.convert("RGB"))))
        folders_dict[folder] = data_item
    return format_data(dataset=folders_dict)


def init_data(images, labels, num_classes, ratio=0.2):
    print("image shape:", images.shape)
    test_size = int(len(images) * ratio)
    x_test = images[:test_size]
    x_train = images[test_size:]
    y_test = labels[:test_size]
    y_train = labels[test_size:]

    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    y_test = to_categorical(y_test, num_classes=num_classes) 
    y_train = to_categorical(y_train, num_classes=num_classes)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    y_test /= 255

    return (x_train, y_train), (x_test, y_test)

def network(x_train,num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
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
                  optimizer=rmsprop(lr=0.0001, decay=1e-6),
                  metrics=["accuracy"])
    return model


def save_model(model, save_dir, model_name):
    save_dir = os.path.join(os.getcwd(), save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print("Saved trained model at %s" % model_path)


if __name__ == "__main__":
    batch_size = 32
    epochs = 20
    images, labels, num_classes = load_data()
    (x_train, y_train), (x_test, y_test) = init_data(images, labels, num_classes)
    model = network(x_train, num_classes)
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, validation_data=(x_test, y_test),
              shuffle=True)
    save_model(model, "saved_models", "sign_language_cnn_trained_mdoels.h5")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print("Test loss: ", loss)
    print("Test accuracy: ", accuracy)
    



