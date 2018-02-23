import tensorflow as tf
import sample_load
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
test_usps_images, test_usps_labels = sample_load.load_usps("./proj3_images/Numerals")

BATCHSIZE = 100

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.int32, [None, 10])

w = tf.Variable(tf.zeros([784, 10]))

#---------------------DEFINITION--------------------------------------------------

def createWeightTensor(shape):

    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def createBiasesTensor(shape):

    return tf.Variable(tf.constant(0.05, shape = shape))

def createConvLayer(weights, x):

    return tf.nn.conv2d(x, weights, strides = [1, 1, 1, 1], padding = 'SAME')

def createMaxPoolLayer(x):

    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def processUSPSLabels(input):

    z = []
    for i in range(input.shape[0]):

        x = [0 for i in range(10)]
        x[input[i]] = 1
        z.append(x)

    return np.array(z)


#-----------------------SETTING THE PARAMETERS-----------------------------------
# first convolutional layer
Weights_conv1 = createWeightTensor([5, 5, 1, 32])
Biases_conv1 = createBiasesTensor([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(createConvLayer(Weights_conv1, x_image) + Biases_conv1)
h_pool1 = createMaxPoolLayer(h_conv1)

# second convolutional layer
Weights_conv2 = createWeightTensor([5,5,32,64])
Biases_conv2 = createBiasesTensor([64])

h_conv2 = tf.nn.relu(createConvLayer(Weights_conv2, h_pool1) + Biases_conv2)
h_pool2 = createMaxPoolLayer(h_conv2)

# densely connected layer
W_dense = createWeightTensor([7 * 7 * 64, 128])
b_dense = createBiasesTensor([128])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_dense = tf.nn.relu(tf.matmul(h_pool2_flat, W_dense) + b_dense)

# drop out

keep_prob = tf.placeholder(tf.float32)
h_dense_drop = tf.nn.dropout(h_dense, keep_prob)

W_class2 = createWeightTensor([128,10])
b_class2 = createBiasesTensor([10])

y_conv = tf.matmul(h_dense_drop, W_class2)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels = y_ ))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#-------------------------------------------TRAINING------------------------------------------

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(6000):
        batch_x, batch_y = mnist.train.next_batch(BATCHSIZE)

        if i % 600 == 0:
            train_accuracy = accuracy.eval(feed_dict = {x: batch_x, y_:batch_y, keep_prob: 1.0})
            print('round %d ---- training accuracy %g for the minibatch' % (i, train_accuracy))

        train_step.run(feed_dict = {x: batch_x, y_:batch_y, keep_prob:0.5})

#-------------------------------------------TESTING--------------------------------------------

    print("MNIST: TEST ACCURACY IS:", sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1}))
    test_usps_labels = processUSPSLabels(test_usps_labels)
    print("USPS: TEST ACCURACY IS:", sess.run(accuracy, feed_dict = {x: test_usps_images, y_: test_usps_labels, keep_prob: 1}))




