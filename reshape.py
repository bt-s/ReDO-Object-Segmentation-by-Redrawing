import tensorflow as tf

z_1 = tf.zeros([25, 1, 1, 32])
z_2 = tf.ones([25, 1, 1, 32])
z = tf.stack((z_1, z_2), axis=1)
print(z.shape)
z_hat1 = tf.ones([25, 32]) * 2
z_hat2 = tf.ones([25, 32]) * 3
z_hat = tf.concat((z_hat1, z_hat2), axis=1)
z_hat = tf.reshape(z_hat, [25, 2, 32])
print(z_hat[:, 1])