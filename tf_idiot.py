import tensorflow as tf

hat_1 = tf.zeros([5, 3])
hat_2 = tf.ones([5, 3])
batch_z_k_hat = tf.concat((hat_1, hat_2), axis=1)
batch_z_k_hat = tf.reshape(batch_z_k_hat, [5, 2, -1])
batch_z_k_hat = tf.transpose(batch_z_k_hat, [1, 0, 2])
batch_z_k_hat = tf.reshape(batch_z_k_hat, [10, -1])
z_k_1 = tf.ones([5, 3]) * 2
z_k_2 = tf.ones([5, 3]) * 3
batch_z_k = tf.concat((z_k_1, z_k_2), axis=0)
print(batch_z_k_hat)
print(batch_z_k)

