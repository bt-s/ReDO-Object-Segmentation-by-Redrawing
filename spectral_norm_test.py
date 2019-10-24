import torch
#!/usr/bin/python3

"""spectral_norm_test.py - Script to test whether our TensorFlow 2.0
                           implementation of spectral normalization is
                           correct.

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chiemelewski-Anders, Mats Steinweg & Bas Straathof"


import torch.nn as nn
from torch.nn.utils import spectral_norm
import tensorflow as tf
import numpy as np
from collections import OrderedDict
from typing import Tuple

import ..network_components

# Suppress TensorFlow info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set PyTorch random seed
torch.manual_seed(0)

# Torch formatted weights
np.random.seed(10)
W = np.random.randn(1, 1, 3, 3)
W_t = torch.from_numpy(W).float()

# TensorFlow formatted weights
np.random.seed(10)
W = np.random.randn(3, 3, 1, 1)
W_tf = tf.convert_to_tensor(W)


def torch_fwd(inp: torch.Tensor , label: torch.Tensor,
        spectral_normalization: bool=False) -> Tuple[torch.Tensor,
                torch.Tensor]:
    """Perform a single foward pass using PyTorch

    Args:
        inp: Input tensor
        label: Label tensor
        spectral_normalization: Whether to specrally normalize the weights

    Returns:
        (out1, out2): The output tensors
    """
    # Create model with or without SN
    if spectral_normalization:
        conv_layer = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1),
                padding=1, bias=False)
        conv_layer.weight.data = W_t
        model = torch.nn.Sequential(OrderedDict([
            ('conv', spectral_norm(conv_layer)),
            ('avg', torch.nn.AvgPool2d(3))]))
    else:
        conv_layer = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1),
                padding=1, bias=False)
        conv_layer.weight.data = W_t
        model = torch.nn.Sequential(OrderedDict([
            ('conv', conv_layer),
            ('avg', torch.nn.AvgPool2d(3))]))

    # Create loss function and SGD optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.zero_grad()
    out1 = model(inp)
    loss = loss_fn(out1, label)
    loss.backward()
    optimizer.step()
    out2 = model(inp)

    return out1, out2


def tf_fwd(inp: tf.Tensor, label: tf.Tensor,
        spectral_normalization: bool=False) -> Tuple[tf.Tensor, tf.Tensor]:
    """Perform a single foward pass using PyTorch

    Args:
        inp: Input tensor
        label: Label tensor
        spectral_normalization: Whether to specrally normalize the weights

    Returns:
        (out1, out2): The output tensors
    """
    # Create model with or without SN
    if spectral_normalization:
        conv = network_components.SpectralNormalization(
                tf.keras.layers.Conv2D(1, kernel_size=(3, 3), strides=(1, 1),
                weights=[W_tf], trainable=False, padding='same',
                input_shape=(3, 3, 1), use_bias=False))
        pool = tf.keras.layers.AveragePooling2D(pool_size=3)
    else:
        conv = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), strides=(1, 1),
                weights=[W_tf], padding='same', input_shape=(3, 3, 1),
                use_bias=False)
        pool = tf.keras.layers.AveragePooling2D(pool_size=3)

    # Create loss function and SGD optimizer
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    _ = conv(inp, training=False)


    with tf.GradientTape() as tape:
        out11 = conv(inp, training=True)
        out1 = pool(out11)
        loss = loss_fn(label, out1)
    gradients = tape.gradient(loss, conv.trainable_variables)

    optimizer.apply_gradients(zip(gradients, conv.trainable_variables))
    out21 = conv(inp, training=True)
    out2 = pool(out21)

    return out1, out2


if __name__ == '__main__':
    # Torch forward pass
    inp_t = torch.ones(1, 1, 3, 3)
    label_t = torch.ones(1, 1, 1, 1)

    out_bu_torch, out_au_torch = torch_fwd(inp_t, label_t,
            spectral_normalization=True)

    # TensorFlow forward pass
    inp_tf = tf.ones([1, 3, 3, 1])
    label_tf = tf.ones([1, 1, 1, 1])

    out_bu_tf, out_au_tf= tf_fwd(inp_tf, label_tf, spectral_normalization=True)

    # Check whether the outputs are the same
    check1 = out_au_torch.tolist()[0][0][0][0] == out_au_tf.numpy()[0][0][0][0]
    check2 = out_bu_torch.tolist()[0][0][0][0] == out_bu_tf.numpy()[0][0][0][0]

    if check1 and check2:
        print('Correct implementation of spectral normalization.')
    else:
        print('Incorrect implementation of spectral normalization.')

