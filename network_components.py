#!/usr/bin/python3

"""network_components.py - Implementation of the components of the various
                           networks

For the NeurIPS Reproducibility Challange and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chiemelewski-Anders, Mats Steinweg & Bas Straathof"


import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, ReLU, \
        Conv2D, MaxPool2D, Softmax, AveragePooling2D
from tensorflow.keras.initializers import orthogonal
from typing import Tuple


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

        # u and weights_sn cannot be initialized yet, since the kernel shape is
        # not known yet
        self.u = None
        self.weights_sn = None


    def normalize_weights(self, training: bool):
        """Normalize the Conv2D layer's weights w.r.t. their spectral norm."""
        # Number of filter kernels in Conv2D layer
        filters = self.layer.weights[0].shape.as_list()[-1]

        # Reshape kernel weights
        W = tf.reshape(self.layer.weights[0], [filters, -1])

        # Compute the singular value approximations
        u, v = self.power_iteration(W)

        # Compute spectral norm
        spectral_norm = tf.matmul(tf.matmul(u, W, transpose_a=True), v)

        # Normalize kernel weights
        self.weights_sn = self.layer.weights[0] / spectral_norm

        if training:
            # Update estimate of singular vector during training
            self.u = u


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
        if self.u is None:
            self.u = tf.Variable(tf.random.normal(
                [self.layer.weights[0].shape.as_list()[-1], 1]),
                trainable=False)

        for _ in range(self.n_power_iterations):
            v = self.normalize_l2(tf.matmul(W, self.u, transpose_a=True))
            u = self.normalize_l2(tf.matmul(W, v))

        return u, v


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


    def call(self, x: tf. Tensor, training: bool) -> tf.Tensor:
        """Perform forward pass of Conv2D layer on first iteration to initialize
        the weights

        Args:
            x:
            training:
        """
        if not self.init:
            _ = self.layer(x)
            self.init = True

        # Normalize weights before performing the standard forward pass of
        # Conv2D layer
        self.normalize_weights(training=training)

        output = self.layer(x)

        return output


# TODO: fix docstring and type-hinting + check for correctness
class SelfAttentionModule(Layer):
    """Self-attention component for GANs"""
    def __init__(self, init_gain: float, output_channels: int,
            key_size: int=None):
        """Class constructor

        Attributes:
            init_gain: Initializer gain for orthogonal initialization
            output_channels: Number of output channels
        """
        super(SelfAttentionModule, self).__init__()

        # Set number of key channels
        if key_size is None:
            self.key_size = output_channels // 8
        else:
            self.key_size = key_size

        # Trainable parameter to control influence of learned attention maps
        self.gamma = tf.Variable(0.0, name='self_attention_gamma')

        # Learned transformation
        self.f = SpectralNormalization(Conv2D(
            filters=self.key_size, kernel_size=(1, 1),
            kernel_initializer=orthogonal(gain=init_gain)))
        self.g = SpectralNormalization(Conv2D(filters=self.key_size,
            kernel_size=(1, 1), kernel_initializer=orthogonal(gain=init_gain)))
        self.h = SpectralNormalization(Conv2D(filters=output_channels,
            kernel_size=(1, 1), kernel_initializer=orthogonal(gain=init_gain)))
        self.out = SpectralNormalization(Conv2D(filters=output_channels,
            kernel_size=(1, 1), kernel_initializer=orthogonal(gain=init_gain)))


    # TODO: fix docstring and type-hinting + check for correctness
    @staticmethod
    def compute_attention(Q, K, V):
        """Compute attention maps from queries, keys and values

        Args:
            Q: Queries
            K: Keys
            V: Values

        Returns:
            Attention map with same shape as input feature maps
        """
        dot_product = tf.matmul(Q, K, transpose_b=True)
        weights = Softmax(axis=2)(dot_product)
        x = tf.matmul(weights, V)

        return x


    # TODO: fix docstring and type-hinting + check for correctness
    def call(self, x, training):
        """

        Args:

        Returns:
        """
        # Height, width, channel
        H, W, C = x.shape.as_list()[1:]

        # Compute query, key and value matrices
        Q = tf.reshape(self.f(x, training), [-1, H * W, self.key_size])
        K = tf.reshape(self.g(x, training), [-1, H * W, self.key_size])
        V = tf.reshape(self.h(x, training), [-1, H * W, C])

        # Compute attention maps
        o = self.compute_attention(Q, K, V)
        o = tf.reshape(o, [-1, H, W, C])
        o = self.out(o, training)

        # Add weighted attention maps to input feature maps
        output = self.gamma * o + x

        return output

