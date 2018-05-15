from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Conv1D
from keras.layers import MaxPooling1D,Dense, Dropout, Activation
from keras.losses import binary_crossentropy
from keras.optimizers import Adam


max_features = 20000
maxlen = 100
embedding_size = 128

kernel_size = 5
filters = 64
pool_size = 4

lstm_output_size = 70

batch_size = 30
epochs = 2 


print("loading data")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), "train sequences")
print(len(x_test), "train sequences")

print("Pad sequences (samples x time)")
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)

print("build model")

model = Sequential()

model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.5))
model.add(Conv1D(filters, kernel_size, padding="valid",
                 activation="relu",strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss=binary_crossentropy,
             optimizer="adam", metrics=["accuracy"])

print("training")
model.fit(x_train, y_train, batch_size=batch_size,
         epochs=epochs, validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("Test score: ", score)
print("Test accuracy: ", acc)


