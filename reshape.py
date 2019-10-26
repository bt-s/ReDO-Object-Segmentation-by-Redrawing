import tensorflow as tf

z_1 = tf.zeros([5, 3])
z_2 = tf.ones([5, 3])
z = tf.stack((z_1, z_2), axis=1)
print(z[:, 1])
z_hat1 = tf.ones([5, 3]) * 2
z_hat2 = tf.ones([5, 3]) * 3
z_hat = tf.concat((z_hat1, z_hat2), axis=1)
z_hat = tf.reshape(z_hat, [5, 2, 3])
print(z_hat[:, 1])