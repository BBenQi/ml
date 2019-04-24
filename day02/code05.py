import tensorflow as tf
import numpy as np


def reshapeTrainData(train):
    num_labels = train.shape[0]
    num_classes = 10
    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + train.ravel()] = 1
    return labels_one_hot


# 准备数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape([60000, 784])
x_test = x_test.reshape([10000, 784])
y_train = reshapeTrainData(y_train)
y_test = reshapeTrainData(y_test)
# 构造模型

x = tf.placeholder(tf.float32, shape=(None, 784))
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, shape=(None, 10))
cross_entropy = tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 开始训练

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
