from day02 import code05
import tensorflow as tf

# 构建Softmax回归模型
# 占位符，None表示任意维度
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placehoder(tf.float32, shape=[None, 10])

# 变量, 将W初始化为零向量，W是784*10的矩阵，因为有784个输入特征，10个输出值
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 初始化变量
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 类别预测与损失函数
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
