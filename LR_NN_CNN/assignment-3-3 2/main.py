import matplotlib.pyplot as plt
import time
import math
import random as rd
import sample_load
import numpy as np
import os

MINIBATCH = 10.
LEARNINGRATE = 0.02
KCLASS = 10


class NNetwork(object):

    def __init__(self, node_numbers, node_connect_mode, function_types, layer_types):

        self.function_types = function_types
        self.layer_types = layer_types

        self.layers = len(node_numbers)
        self.sizes = node_numbers

        self.activitation = []
        self.output = [np.array([1]) for i in range(self.layers)]
        self.input = [np.array([1]) for i in range(self.layers)]

        self.connect_weights = []
        self.connect_bias = []

        self.temp_weights = []    # save the sum of updated weights for minibatches
        self.temp_bias = []       # save the sum of updated bias for minibatches


        self.connect_weights.append(np.ndarray((1,1))) # for
        self.connect_bias.append(np.ndarray((1,1)))

        self.temp_weights.append(np.ndarray((1, 1)))  # for
        self.temp_bias.append(np.ndarray((1, 1)))

        self.node_connect_mode = node_connect_mode

        self.n = 0

        for i in range(len(node_numbers[1:])):

            self.connect_weights.append(np.random.randn(node_numbers[i+1], node_numbers[i]))
            self.connect_bias.append(np.random.randn(node_numbers[i+1], 1))

            self.temp_weights.append(np.zeros((node_numbers[i + 1], node_numbers[i])))
            self.temp_bias.append(np.zeros((node_numbers[i + 1], 1)))


    def activitateFunction(self, input, type):

        if 'SIGMOID' == type:
            return self.nd_sigmoid(input)
        if 'SOFTMAX' == type:
            return self.nd_softmax(input)

    def nd_sigmoid(self, a):
        a /= 1000.
        return 1./(np.exp(-a) + 1.)

    def nd_softmax(self, ax):

        return np.exp(ax)/np.sum(np.exp(ax))


    def feedFowardNN(self, input):

        self.output[0] = input
        w = self.connect_weights
        b = self.connect_bias

        for i in range(self.layers - 1):  # ignore the input layer
            # i + 1
            s = np.dot(w[i+1], self.output[i])
            s.shape = (s.shape[0],1)
            a = s + b[i+1]
            # print(a)
            self.input[i+1] = a
            self.output[i+1] = self.activitateFunction(a, self.function_types[i])

    def accumulate(self):

        for i in range(self.layers-1):

            self.temp_weights[i+1] = np.zeros(self.temp_weights[i+1].shape)
            self.temp_bias[i+1] = np.zeros(self.temp_bias[i+1].shape)

    def derive(self, input):

        x = self.nd_sigmoid(input)
        return x*(1-x)

    def backPropagate(self, target):

        # target.shape = (target.shape[0],1)
        em = self.output[-1]
        em[target] -= 1
        t = [0,1,2]
        t.reverse()

        for m in t:

            if m == 0:
                break
            if 'OUTPUT' == self.layer_types[m]:
                k = np.matrix(em) * (np.matrix(self.output[m-1]).T)
                k = np.array(k)
                self.temp_weights[m] += k
                self.temp_bias[m] += em

            if 'HIDDEN' == self.layer_types[m]:

                haj = self.derive(self.input[m])
                wkj = np.matrix(self.connect_weights[m+1]).T * np.matrix(em)
                wkj = haj*np.array(wkj)
                wkj = np.matrix(wkj)

                wkj = wkj.T

                temp = np.matrix(self.output[m-1]).T * wkj

                self.temp_weights[m] += np.array(temp.T)
                self.temp_bias[m] += np.array(wkj.T)

    def updateParameters(self):
        # print("No. ", self.n, " minibatch")

        for i in range(self.layers - 1):

            self.connect_weights[i+1] -= ((self.temp_weights[i+1]*LEARNINGRATE)/MINIBATCH)
            self.connect_bias[i+1] -= ((self.temp_bias[i+1]*LEARNINGRATE)/MINIBATCH)

        self.accumulate()
        self.n += 1


    def training(self, train_data, train_label):

        k = 0
        for i in range(60000):

            self.feedFowardNN(train_data[i])
            self.backPropagate(train_label[i])

            k += 1

            if 10 == k:
                self.updateParameters()
                k = 0


    def testing(self, test_data, test_label):

        error = 0
        lenx = test_data.shape[0]
        for i in range(lenx):
            self.feedFowardNN(test_data[i])

            x = self.output[-1]
            MAX = 0.
            MAX_j = -1
            for j in range(x.shape[0]):
                if x[j] > MAX:
                    MAX_j = j
                    MAX = x[j]

            if MAX_j != test_label[i]:
                error += 1

        return (error/lenx)

#---------------------EXECUTE PART--------------------------------------------

while True:
    option = input("Please input option: 1 for Regression, 2 for hidden layer neural network, 3 for CNN")
    if option not in ['1','2','3']:
        continue
    else:
        if '1' == option:
            os.system("python3 ./LogRegression.py")
            break
        if '2' == option:

            test_images, test_lables = sample_load.load_mnist("./", "t10k")
            train_images, train_lables = sample_load.load_mnist("./", "train")
            test_usps_images, test_usps_lables = sample_load.load_usps("./proj3_images/Numerals")
            print(train_images[0].shape)

            a = NNetwork([784,50,10],'FULL',['SIGMOID','SOFTMAX'],['INPUT','HIDDEN','OUTPUT'])
            round = 0
            while True:
                print("No.", round, " training")
                for i in range(5):
                    a.training(train_images, train_lables)
                err = a.testing(test_images,test_lables)

                print("MNIST: error rate is:", err)

                if err >= 0.057:
                    round += 1
                    continue
                else:
                    break

            err2 = a.testing(test_usps_images, test_usps_lables)
            print("USPS: error rate is:", err2)
            break
        if '3' == option:
            os.system("python3 ./tf-cnn.py")
            break