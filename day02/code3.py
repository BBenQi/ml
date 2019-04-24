import tensorflow as tf

# 创建变量，初始化为标量0，用来计数
state = tf.Variable(0, name="counter")

# 创建OP，使state+1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图
init = tf.global_variables_initializer()

# 运行图

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))

    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
