#!/usr/bin/python3

"""discriminator.py - Implementation of the discriminator network and its
                      components

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chmielewski-Anders, Mats Steinweg & Bas Straathof"


import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, ReLU, \
        Conv2D, MaxPool2D, Softmax, GlobalAveragePooling2D
from tensorflow.keras.initializers import orthogonal
from typing import Union, Tuple

from network_components import SelfAttentionModule, SpectralNormalization


class ResidualBlock(Layer):
    """Residual computation block with down-sampling option and variable number
    of output channels."""
    def __init__(self, init_gain: float, stride: Union[int, Tuple[int, int]],
            output_channels: int):
        """Class constructor

        Args:
            init_gain: Initializer gain for orthogonal initialization
            stride: Stride of the convolutional layers
            output_channels: Number of output channels
        """
        super(ResidualBlock, self).__init__()

        # Perform 1x1 convolutions on the identity to adjust the number of
        # channels to the output of the residual pipeline
        self.process_identity = SpectralNormalization(Conv2D(
            filters=output_channels, kernel_size=(1, 1), strides=stride,
            kernel_initializer=orthogonal(gain=init_gain)))

        # Residual pipeline
        self.conv_1 = SpectralNormalization(Conv2D(
            filters=output_channels, kernel_size=(3, 3), strides=stride,
            padding='same', kernel_initializer=orthogonal(gain=init_gain)))
        self.relu = ReLU()
        self.conv_2 = SpectralNormalization(Conv2D(filters=output_channels,
            kernel_size=(3, 3), padding='same',
            kernel_initializer=orthogonal(gain=init_gain)))

    def call(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        """Perform call of residual block layer call

        Args:
            x: Input to the residual block
            training: Whether we are training
        """
        # Save identity
        identity = x

        # Pass input through pipeline
        x = self.relu(x)
        x = self.conv_1(x, training)
        x = self.relu(x)
        x = self.conv_2(x, training)

        if x.shape != identity.shape:
            identity = self.process_identity(identity, training)

        # Skip-connection
        x += identity

        return x


class Discriminator(Model):
    """Discriminator model for the GAN"""
    def __init__(self, init_gain: float):
        """Class constructor

        ArgS:
            init_gain: Initializer gain for orthogonal initialization
        """
        super(Discriminator, self).__init__()

        # Set model's name
        self.model_name = 'Discriminator'

        # Set ReLU
        self.relu = ReLU()

        # Input residual down-sampling block
        self.block_1 = ResidualBlock(init_gain=init_gain, output_channels=64,
                stride=(2, 2))

        # Self-attention module
        self.block_2 = SelfAttentionModule(init_gain=init_gain,
                output_channels=64)

        # Sequence of residual down-sampling blocks
        self.res_block_2 = ResidualBlock(init_gain=init_gain,
                output_channels=64, stride=(2, 2))
        self.res_block_3 = ResidualBlock(init_gain=init_gain,
                output_channels=128, stride=(2, 2))
        self.res_block_4 = ResidualBlock(init_gain=init_gain,
                output_channels=256, stride=(2, 2))
        self.res_block_5 = ResidualBlock(init_gain=init_gain,
                output_channels=512, stride=(2, 2))
        self.res_block_6 = ResidualBlock(init_gain=init_gain,
                output_channels=1024, stride=(1, 1))

        # Spatial sum pooling
        self.block_4 = GlobalAveragePooling2D()

        # Dense classification layer
        self.block_5 = Dense(units=1,
                kernel_initializer=orthogonal(gain=init_gain))

    def call(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        """Call the Discriminator network

        Args:
            x: Input to the residual block
            training: Whether we are training
        """
        x = self.block_1(x, training)
        x = self.block_2(x, training)
        x = self.res_block_2(x, training)
        x = self.res_block_3(x, training)
        x = self.res_block_4(x, training)
        x = self.res_block_5(x, training)
        x = self.res_block_6(x, training)
        x = self.relu(x)
        x = self.block_4(x) * x.shape[1] * x.shape[2]
        x = self.block_5(x)

        return x

