import tensorflow as tf
from normalizations import InstanceNormalization
from tensorflow.keras.layers import LayerNormalization

in_norm = InstanceNormalization()
la_norm = LayerNormalization(axis=(1, 2), center=True, scale=True)

inp1 = tf.random.normal([25, 128, 128, 3])
inp2 = tf.random.normal([25, 128, 128, 3])

out1 = in_norm(inp1)
out2 = la_norm(inp1)

print(in_norm.gamma)
print(in_norm.beta)
print(la_norm.variables)
tf.assert_equal(out1, out2)