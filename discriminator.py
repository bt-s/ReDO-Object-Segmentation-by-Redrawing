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
        Conv2D, MaxPool2D, Softmax, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.initializers import orthogonal
from typing import Union, Tuple

from network_components import SelfAttentionModule, SpectralNormalization, \
        ResidualBlock


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
                stride=(1, 1), first_block=True)

        # Self-attention module
        self.block_2 = SelfAttentionModule(init_gain=init_gain,
                output_channels=64)

        # Sequence of residual down-sampling blocks
        self.res_block_2 = ResidualBlock(init_gain=init_gain,
                output_channels=128, stride=(1, 1))
        self.res_block_3 = ResidualBlock(init_gain=init_gain,
                output_channels=256, stride=(1, 1))
        self.res_block_4 = ResidualBlock(init_gain=init_gain,
                output_channels=512, stride=(1, 1))
        self.res_block_5 = ResidualBlock(init_gain=init_gain,
                output_channels=1024, stride=(1, 1))
        self.res_block_6 = ResidualBlock(init_gain=init_gain,
                output_channels=2048, stride=(1, 1), downsample=False)

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
        x = self.block_4(x) * x.shape[1] * x.shape[2]
        x = self.block_5(x)

        return x

