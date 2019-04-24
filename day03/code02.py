import tensorflow as tf

W = tf.Variable(tf.random_normal([784, 10], stddev=0.5))
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    result = sess.run(W)
    print(result[0])
