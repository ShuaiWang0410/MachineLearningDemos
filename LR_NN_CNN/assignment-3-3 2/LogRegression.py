# import matplotlib.pyplot as plt
import time
import math
import random as rd
import sample_load
import numpy as np

test_images, test_lables = sample_load.load_mnist("./", "t10k")
train_images, train_lables = sample_load.load_mnist("./", "train")
test_usps_images, test_usps_lables = sample_load.load_usps("./proj3_images/Numerals")

weights = ''
bias = ''
temp_weights = ''
temp_bias = ''

MINIBATCH = 10
LEARNINGRATE = 0.0005
KCLASS = 10

def softmax(a):
    a /= 1000.
    r = np.exp(a)/np.sum(np.exp(a))
    return r


def setup(w_numbers, class_numbers):

    global weights
    global bias
    global temp_weights
    global temp_bias

    weights = np.random.randn(class_numbers, w_numbers)
    bias = np.random.randn(class_numbers, 1)

    temp_weights = np.zeros((class_numbers, w_numbers))
    temp_bias = np.zeros((class_numbers, 1))


def inModel(a):

    w = weights
    b = bias

    s = np.dot(w, a)
    s.shape = (s.shape[0], 1)
    r = s + b

    return softmax(r)

def accWeights(target, r, a):


    t_weights = temp_weights
    t_bias =  temp_bias
    target.shape = (10,1)
    s = r - target
    w = s*a
    t_weights += w
    t_bias += s

def updateWeights():

    tw = temp_weights
    tb = temp_bias

    w = weights
    b = bias

    w -= (LEARNINGRATE * tw/MINIBATCH)
    b -= (LEARNINGRATE * tb/MINIBATCH)

    clear()


def clear():

    global temp_weights
    global temp_bias

    temp_weights = np.zeros(temp_weights.shape)
    temp_bias = np.zeros(temp_bias.shape)


def train(train_images, train_lables):
    k = 0

    for i in range(train_images.shape[0]):

        r = inModel(train_images[i])
        a = np.array([0. for i in range(KCLASS)])
        a[train_lables[i]] = 1.
        accWeights(a, r, train_images[i])

        k += 1

        if MINIBATCH == k:
            updateWeights()
            k = 0


def test(test_images, test_lables):
    error = 0
    lenx = test_images.shape[0]
    for i in range(lenx):
        r = inModel(test_images[i])
        # print(r)

        MAX = 0.
        MAX_j = -1
        for j in range(r.shape[0]):
            if r[j] > MAX:
                MAX_j = j
                MAX = r[j]

        if MAX_j != test_lables[i]:
            error += 1
    return error/lenx

# EXECUTE PART



setup(784, KCLASS)

n = 0
while True:

    print("No. ", n, " training")
    train(train_images, train_lables)
    err = test(test_images, test_lables)
    print("MNIST: error rate is ", err)

    if (err) >= 0.0795:
        n += 1
    else:
        break

err2 = test(test_usps_images, test_usps_lables)
print("USPS: error rate is ", err2)



















