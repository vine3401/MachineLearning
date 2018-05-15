import tensorflow as tf

x = tf.constant(1, dtype=tf.float32)
y = tf.constant(2, dtype=tf.float32)
z = tf.Variable(3, dtype=tf.float32, trainable=False)
x_add_y = x + y