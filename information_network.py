#!/usr/bin/python3

"""information_network.py - Implementation of the information network and its
                            components

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chmielewski-Anders, Mats Steinweg & Bas Straathof"


import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.initializers import orthogonal
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Conv2D, \
        MaxPool2D, Softmax, GlobalAveragePooling2D

from network_components import ResidualBlock, SelfAttentionModule


class InformationConservationNetwork(Model):
    """Network to recover the latent z-values"""
    def __init__(self, init_gain: float, n_classes: int, n_output: int):
        """Class constructor

        Attributes:
            init_gain: Initializer for the kernel weights
            n_classes: Number of classes in the training scheme. Corresponds
                       to the number of network heads
            n_output: Dimensionality of the output data
        """
        super(InformationConservationNetwork, self).__init__()

        self.model_name = 'Information_Network'
        self.n_classes = n_classes

        # ReLU
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
                output_channels=1024, stride=(1, 1), downsample=False)

        # Spatial sum pooling
        self.block_4 = GlobalAveragePooling2D()

        # Dense classification layers
        self.final_layer = Dense(units=n_output*self.n_classes,
                kernel_initializer=orthogonal(gain=init_gain))

    def call(self, x: tf.Tensor, training: bool):
        """Applies the information conservation network

        Args:
            x: Input batch of shape (n, 128, 128, 3)
            training: Whether we are in the training phase
        """
        # Perform forward pass
        x = self.block_1(x, training)
        x = self.block_2(x, training)
        x = self.res_block_2(x, training)
        x = self.res_block_3(x, training)
        x = self.res_block_4(x, training)
        x = self.res_block_5(x, training)
        x = self.res_block_6(x, training)
        x = self.block_4(x) * x.shape[1] * x.shape[2]
        x = self.final_layer(x)
        x = tf.reshape(x, [x.shape[0], self.n_classes, -1])

        return x

if __name__ == '__main__':

    tf1 = tf.ones([25, 128, 128, 3])
    #inform = InformationConservationNetwork(init_gain=0.8, n_classes=2, n_output=32)
    #out = inform(tf1, training=True)

    tf2 = tf.random.normal([20, 2, 1, 1, 32], 0, 1)
    tf21 = tf2[:, :, 0, 0, :]
    tf3 = tf.random.normal([20, 2, 32], 0, 3)

    tf4 = (tf2 - tf3)
    loss = tf.reduce_mean(tf4 * tf4)
    tf41 = (tf21 - tf3)
    loss1 = tf.reduce_mean(tf41 * tf41)
    print(tf4.shape)
    print(loss)
    print(loss1)