#!/usr/bin/python3

"""test_instance_norm.py - Script to test whether our TensorFlow 2.0
                           implementation of instance normalization is
                           correct. The test executes a single forward pass
                           and compares the outputs against the PyTorch
                           implementation (see: https://pytorch.org/docs/stable/
                           _modules/torch/nn/InstanceNorm2D.html).

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chiemelewski-Anders, Mats Steinweg & Bas Straathof"


import tensorflow as tf
from network_components import InstanceNormalization
from tensorflow.keras.layers import LayerNormalization
import torch.nn as nn
import torch
import numpy as np

# Torch formatted weights
np.random.seed(10)
inp_t = np.random.randn(20, 3, 128, 128)
inp_t = torch.from_numpy(inp_t).float()

# TensorFlow formatted weights
np.random.seed(10)
inp_tf = np.random.randn(20, 3, 128, 128)
inp_tf = tf.transpose(inp_tf, [0, 2, 3, 1])
inp_tf = tf.convert_to_tensor(inp_tf)

print(inp_tf[:3, 0, 0, 0:3])
print(inp_t[:3, :3, 0, 0])

in_norm = InstanceNormalization()
in_norm_torch = nn.InstanceNorm2d(3, affine=False)

out_tf = in_norm(inp_tf)
out_torch = in_norm_torch(inp_t)

print(out_tf[0:3, 0, 0, 0:3])
print(out_torch[:3, :3, 0, 0])
out_tf = tf.transpose(out_tf, [0, 3, 1, 2])

if np.allclose(out_tf.numpy(), out_torch.numpy()):
    print('Instance Norm test passed!')
