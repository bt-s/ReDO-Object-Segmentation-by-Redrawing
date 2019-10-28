import tensorflow as tf

tf1 = tf.ones([1, 5, 5, 1])
tf2 = tf.ones([1, 5, 5, 1]) * 2
tf3 = tf.ones([1, 5, 5, 1]) * 3
tf4 = tf.concat((tf1, tf2, tf3), axis=3)
print(tf4)
mask = tf.zeros([1, 5, 5, 1])

tf4 *= mask

print(tf4)