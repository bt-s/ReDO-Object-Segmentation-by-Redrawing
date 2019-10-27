#!/usr/bin/python3

"""spectral_norm_test.py - Script to test whether our TensorFlow 2.0
                           implementation of spectral normalization is
                           correct. The test executes a single forward pass
                           and compares the outputs against the PyTorch
                           implementation (see: https://pytorch.org/docs/stable/
                           _modules/torch/nn/utils/spectral_norm.html).

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chiemelewski-Anders, Mats Steinweg & Bas Straathof"


import torch
from torch.nn.utils import spectral_norm
import tensorflow as tf
import numpy as np
from collections import OrderedDict
from typing import Tuple

from tensorflow.keras.layers import Conv2D, Layer
from tensorflow.keras.initializers import orthogonal

class SpectralNormalization(Layer):
    """Spectral normalization layer to wrap around a Conv2D Layer. Kernel
    weights are normalized before each forward pass."""
    def __init__(self, layer: Conv2D, n_power_iterations: int=1):
        """Class constructor

        Attributes:
            layer: Conv2D object
            n_power_iterations: Number of power iterations
        """
        super(SpectralNormalization, self).__init__()

        self.layer = layer
        self.n_power_iterations = n_power_iterations

        # Conv2D layer's weights haven't been initialized yet
        self.init = False

        # Initialize u
        self.u = super().add_weight(name='u', shape=[self.layer.filters, 1],
            initializer=tf.initializers.RandomNormal, trainable=False)

    def call(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        """Perform forward pass of Conv2D layer on first iteration to initialize
        the weights

        Args:
            x:
            training:
        """
        # Perform forward pass of Conv2D layer on first iteration to initialize
        # the weights. Introduce 'kernel_orig' as trainable variables
        if not self.init:
            # Initialize Conv2D layer
            _ = self.layer(x)
            self.layer.kernel_orig = self.layer.add_weight('kernel_orig',
                                                           self.layer.kernel.shape, trainable=True)

            weights = self.layer.get_weights()
            # Set 'kernel_orig' to network's weights. 'kernel_orig' will be
            # updated, 'kernel' will be normalized and used in the forward pass
            if len(weights) == 2:
                # Conv layer without bias
                self.layer.set_weights([weights[0], weights[0]])
                tf.assert_equal(self.layer.weights[0], self.layer.weights[1])

            else:
                # Conv layer with bias
                self.layer.set_weights([weights[0], weights[1],
                                        weights[0]])
                tf.assert_equal(self.layer.weights[0], self.layer.weights[2])

            # SN layer initialized
            self.init = True

        # Normalize weights
        W_sn = self.normalize_weights(training=training)

        # assign normalized weights to kernel for forward pass
        self.layer.kernel = W_sn

        # perform forward pass of Conv2d layer
        output = self.layer(x)

        return output

    def normalize_weights(self, training: bool):
        """Normalize the Conv2D layer's weights w.r.t. their spectral norm."""

        # Number of filter kernels in Conv2D layer
        filters = self.layer.weights[0].shape.as_list()[-1]

        # Get original weights
        W_orig = self.layer.kernel_orig

        # Reshape kernel weights
        W_res = tf.reshape(W_orig, [filters, -1])

        # Compute spectral norm and singular value approximation
        spectral_norm, u = self.power_iteration(W_res)

        # Normalize kernel weights
        W_sn = W_orig / spectral_norm

        if training:
            # Update estimate of singular vector during training
            self.u.assign(u)

        return W_sn

    def power_iteration(self, W: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute approximate spectral norm.

        Note: According to the paper n_power_iterations= 1 is sufficient due
              to updated u.

        Args:
            W: Reshaped kernel weights | shape: [filters, N]

        Returns:
            Approximate spectral norm and updated singular vector
            approximation.
        """
        for _ in range(self.n_power_iterations):
            v = self.normalize_l2(tf.matmul(W, self.u, transpose_a=True))
            u = self.normalize_l2(tf.matmul(W, v))
            spectral_norm = tf.matmul(tf.matmul(u, W, transpose_a=True), v)

        return spectral_norm, u

    @staticmethod
    def normalize_l2(v: tf.Tensor, epsilon: float=1e-12) -> tf.Tensor:
        """Normalize input matrix w.r.t. its euclidean norm

        Args:
            v: Input matrix
            epsilon: Small epsilon to avoid division by zero

        Returns:
            l2-normalized input matrix
        """
        return v / (tf.math.reduce_sum(v**2)**0.5 + epsilon)

if __name__ == '__main__':

    tf.random.set_seed(0)

    s_norm = SpectralNormalization(Conv2D(filters=5, kernel_size=(3, 3), kernel_initializer=orthogonal(gain=0.8)))
    inp = tf.random.normal([1, 32, 32, 1])
    _ = s_norm(inp, training=True)
    print([var.name for var in s_norm.layer.trainable_variables])
