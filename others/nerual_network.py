import random
import gzip
import pickle
import numpy as np


def sigmoid(z):
<<<<<<< HEAD
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
=======
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
>>>>>>> 484296a79105d6993b25afdd6fa716e4bb9b068b


def load_data():
    f = gzip.open("../datasets/mnist.pkl.gz", "rb")
    training_data, validation_data, test_data = pickle.loads(
        f.read(), encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


class NetWork(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
<<<<<<< HEAD
        if test_data:
            n_test = len(test_data)
=======
        if test_data: n_test = len(test_data)
>>>>>>> 484296a79105d6993b25afdd6fa716e4bb9b068b
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
<<<<<<< HEAD
                training_data[k:k+mini_batch_size]
=======
                training_data[k:k + mini_batch_size]
>>>>>>> 484296a79105d6993b25afdd6fa716e4bb9b068b
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                right = self.evaluate(test_data)
<<<<<<< HEAD
                print("Epoch {0}: {1} / {2}  ac: {3}%".format(j,
                                                              right, n_test, right * 100 / n_test))
=======
                print("Epoch {0}: {1} / {2}  ac: {3}%".format(j, right, n_test, right * 100 / n_test))
>>>>>>> 484296a79105d6993b25afdd6fa716e4bb9b068b
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
<<<<<<< HEAD
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
=======
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
>>>>>>> 484296a79105d6993b25afdd6fa716e4bb9b068b
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
<<<<<<< HEAD
            z = np.dot(w, activation)+b
=======
            z = np.dot(w, activation) + b
>>>>>>> 484296a79105d6993b25afdd6fa716e4bb9b068b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
<<<<<<< HEAD
            sigmoid_prime(zs[-1])
=======
                sigmoid_prime(zs[-1])
>>>>>>> 484296a79105d6993b25afdd6fa716e4bb9b068b
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
<<<<<<< HEAD
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
=======
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
>>>>>>> 484296a79105d6993b25afdd6fa716e4bb9b068b
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    def cost_derivative(self, output_activations, y):
<<<<<<< HEAD
        return (output_activations-y)
=======
        return (output_activations - y)
>>>>>>> 484296a79105d6993b25afdd6fa716e4bb9b068b


training_data, validation_data, test_data = load_data_wrapper()
net = NetWork([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
