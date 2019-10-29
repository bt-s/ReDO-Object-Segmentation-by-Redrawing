import tensorflow as tf

tf1 = tf.random.normal([3, 1])
tf2 = tf.random.normal([3, 1])
indices = tf.range(start=0, limit=tf.shape(tf1)[0], dtype=tf.int32)
shuffled_indices = tf.random.shuffle(indices)

shuffled_x = tf.gather(tf1, shuffled_indices)
shuffled_y = tf.gather(tf2, shuffled_indices)
print(tf1)
print(shuffled_x)
print(tf2)
print(shuffled_y)

tf.random.set_seed(0)
z1 = tf.random.normal([1, 5])
tf.random.set_seed(0)
z2 = tf.random.normal([1, 5])
print(z1)
print(z2)


noise1 = tf.ones([20, 1, 1, 32])
noise2 = tf.ones([20, 1, 1, 32]) * 2
noise = tf.stack((noise1, noise2), axis=1)
print(noise[:, 0])

h = tf.random.normal([2, 2])
sz = tf.math.count_nonzero(h)
print(h)
print(sz)

