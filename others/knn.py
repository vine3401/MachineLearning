import operator

import numpy as np


def autoNorm(dataset):
    attributes = dataset.shape[1]
    for i in range(0, attributes):
        dataset[:, i] = (dataset[:, i] - dataset[:, i].mean()) / dataset[:, i].std()
    return dataset

def knn(dataset, label, k, inX):
    lines = dataset.shape[0]
    distance = np.sqrt(((np.tile(inX, (lines, 1)) - dataset)**2).sum(axis=1))
    sortedDistance = distance.argsort()[0:k]
    classLabel = {}
    for item in range(k):
        l = label[sortedDistance[item]]
        classLabel[l] = classLabel.get(l, 0)+1
    key = sorted(classLabel.items(), key=operator.itemgetter(1), reverse=True)
    return key[0][0]


def test(data, label, test_ratio=0.2):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_data = data[test_indices]
    train_label = label[train_indices]
    error = 0
    for i in test_indices:
        pred = knn(train_data, train_label, 20, data[i][0:3])
        if pred != label[i]:
            error += 1
    print(error, test_set_size)
    print(error / test_set_size)


source = np.loadtxt("../datasets/datingTestSet2.txt")

data = autoNorm(source[:, 0:3])
label = source[:, -1]
label_prediction = knn(data, label, 3, [1,2,3])
test(data, label)


