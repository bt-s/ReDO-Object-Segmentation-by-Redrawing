#!/usr/bin/python3

"""network_components.py - Implementation of the components of the various
                           networks

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chmielewski-Anders, Mats Steinweg & Bas Straathof"


import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, ReLU, \
        Conv2D, MaxPool2D, Softmax, AveragePooling2D
from tensorflow.keras.initializers import orthogonal
from tensorflow.keras.constraints import Constraint
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

        # u and W_orig cannot be initialized yet, since the kernel shape is
        # not known yet
        self.u = None
        self.W_orig = None


    def normalize_weights(self, training: bool):
        """Normalize the Conv2D layer's weights w.r.t. their spectral norm."""
        # Number of filter kernels in Conv2D layer
        filters = self.layer.weights[0].shape.as_list()[-1]

        # Store the original weights
        W_orig = self.layer.kernel_orig

        # Reshape kernel weights
        W_res = tf.reshape(W_orig, [filters, -1])

        # Compute spectral norm and singular value approximation
        spectral_norm, u = self.power_iteration(W_res)

        # Normalize kernel weights
        W_sn = W_orig / spectral_norm

        if training:
            # Update estimate of singular vector during training
            self.u = u

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
        if self.u is None:
            self.u = tf.Variable(tf.random.normal(
                [self.layer.weights[0].shape.as_list()[-1], 1]),
                trainable=False)

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


    def call(self, x: tf. Tensor, training: bool) -> tf.Tensor:
        """Perform forward pass of Conv2D layer on first iteration to initialize
        the weights

        Args:
            x:
            training:
        """
        if not self.init:
            _ = self.layer(x)
            self.layer.kernel_orig = self.add_weight('kernel_orig',
                    self.layer.kernel.shape, trainable=True)
            weights = self.layer.get_weights()
            self.layer.set_weights([tf.identity(weights[0]), weights[0]])
            self.init = True

        # Normalize weights before performing the standard forward pass of
        # Conv2D layer
        W_sn = self.normalize_weights(training=training)
        self.layer.kernel = W_sn
        output = self.layer(x)

        return output


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


    def compute_attention(self, x: tf.Tensor, train: bool) -> tf.Tensor:
        """Compute attention maps

        Args:
            x: Input to the residual block
            training: Whether we are training

        Returns:
            Attention map with same shape as input feature maps
        """
        # Height, width, channel
        h, w, c = x.shape.as_list()[1:]

        fx = tf.reshape(self.f(x, train), [-1, h * w, self.key_size])
        gx = tf.reshape(self.g(x, train), [-1, h * w, self.key_size])
        s = tf.matmul(fx, gx, transpose_b=True)

        beta = Softmax(axis=2)(s)

        hx = tf.reshape(self.h(x, train), [-1, h * w, c])

        interim = tf.matmul(beta, hx)
        interim = tf.reshape(interim, [-1, h, w, c])
        o = self.out(interim, train)

        return o


    def call(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        """Perform call of attention layer

        Args:
            x: Input to the residual block
            training: Whether we are training
        """
        o = self.compute_attention(x, training)
        y = self.gamma * o + x

        return y

