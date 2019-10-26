import tensorflow as tf
from normalizations import InstanceNormalization
from tensorflow.keras.layers import LayerNormalization
import torch.nn as nn
import torch
import numpy as np

# Torch formatted weights
np.random.seed(10)
inp_t = np.random.randn(25, 3, 128, 128)
inp_t = torch.from_numpy(inp_t).float()

# TensorFlow formatted weights
np.random.seed(10)
inp_tf = np.random.randn(25, 128, 128, 3)
inp_tf = tf.convert_to_tensor(inp_tf)

in_norm = InstanceNormalization()
la_norm = LayerNormalization(axis=(1, 2), center=True, scale=True)
in_norm_torch = nn.InstanceNorm2d(3, affine=True)

out_tf = in_norm(inp_tf)
out_torch = in_norm_torch(inp_t)
out2 = la_norm(inp_tf)

print(out_tf)
print(out_torch)
