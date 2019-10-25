#!/usr/bin/python3

"""segmentation_network.py - Implementation of the segmentation network and its
                             components

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chmielewski-Anders, Mats Steinweg & Bas Straathof"


import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization, ReLU, \
        UpSampling2D, Softmax, AveragePooling2D, ReLU
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import orthogonal
from tensorflow.keras.regularizers import L1L2
from typing import Union, Tuple


class ConvolutionalBlock(Model):
    """Computational block consisting of a 2D Convolutional layer followed by
    an Instance Normalization layer and ReLU activation."""
    def __init__(self, filters: int, kernel_size: Tuple[int, int], padding: str,
            stride: Union[int, Tuple[int, int]], init_gain: float,
            weight_decay: float):
        """Class constructor

        Attributes:
            filters: Number of filters in the convolutional layer
            kernel_size: Kernel size of the convolutional layer
            padding: Padding of the convolutional layer
            stride: Stride of the convolutional layer
            init_gain: Initializer gain for orthogonal initialization
            weight_decay: Weight decay constant
        """
        super(ConvolutionalBlock, self).__init__()

        self.conv_block = Sequential()
        self.conv_block.add(LayerNormalization(axis=(1, 2),
            center=True, scale=True))
        self.conv_block.add(ReLU())
        self.conv_block.add(Conv2D(filters=filters, kernel_size=kernel_size,
            padding=padding, strides=stride, use_bias=False,
            kernel_initializer=orthogonal(gain=init_gain),
            kernel_regularizer=L1L2(l2=weight_decay)))

    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Perform call of convolutional block

        Args:
            x: Input to the convolutional block
        """
        x = self.conv_block(x)

        return x


class PPM(Model):
    """Pyramid Pooling Module - extracts features at four different scales and
    fuses them together."""
    def __init__(self, input_shape: Tuple[int, int, int], init_gain: float,
            weight_decay: float):
        """Class constructor

        Attributes:
            init_gain: Initializer gain for orthogonal initialization
            input_shape: Shape of the input feature maps (W, H, C)
            weight_decay: Multiplicative factor for l2 weight regularization
        """
        super(PPM, self).__init__()

        if not len(input_shape) == 3:
            raise ValueError("Input parameter <input_dim> must be of shape "
                    "(W, H, C).")

        # Scale 1 (1x1 Output)
        pool_size_1 = (input_shape[0] // 1, input_shape[1] // 1)
        self.avg_pool_1 = AveragePooling2D(pool_size_1)
        self.conv_1 = Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                kernel_initializer=orthogonal(gain=init_gain),
                kernel_regularizer=L1L2(l2=weight_decay))
        self.upsample_1 = UpSampling2D(size=pool_size_1,
                interpolation='bilinear')

        # Scale 2 (2x2 Output)
        pool_size_2 = (input_shape[0] // 2, input_shape[1] // 2)
        self.avg_pool_2 = AveragePooling2D(pool_size_2)
        self.conv_2 = Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                kernel_initializer=orthogonal(gain=init_gain),
                kernel_regularizer=L1L2(l2=weight_decay))
        self.upsample_2 = UpSampling2D(size=pool_size_2,
                interpolation='bilinear')

        # Scale 3 (4x4 Output)
        # TODO: Maybe change this to // 3
        pool_size_3 = (input_shape[0] // 4, input_shape[1] // 4)
        self.avg_pool_3 = AveragePooling2D(pool_size_3)
        self.conv_3 = Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                kernel_initializer=orthogonal(gain=init_gain),
                kernel_regularizer=L1L2(l2=weight_decay))
        self.upsample_3 = UpSampling2D(size=pool_size_3,
                interpolation='bilinear')

        # Scale 4 (8x8 Output)
        # TODO: Maybe change this to // 6
        # Note: The upsampling issue should be fixed
        pool_size_4 = (input_shape[0] // 8, input_shape[1] // 8)
        self.avg_pool_4 = AveragePooling2D(pool_size_4)
        self.conv_4 = Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                kernel_initializer=orthogonal(gain=init_gain),
                kernel_regularizer=L1L2(l2=weight_decay))
        self.upsample_4 = UpSampling2D(size=pool_size_4,
                interpolation='bilinear')

        # Final up-sampling
        self.upsample_final = UpSampling2D(size=(2, 2),
                interpolation='bilinear')
    

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Perform call of PPM block

        Args:
            x: Input to the PPM block
        """
        # Scale 1
        x_1 = self.avg_pool_1(x)
        x_1 = self.conv_1(x_1)
        x_1 = ReLU()(x_1)
        x_1 = self.upsample_1(x_1)

        # Scale 2
        x_2 = self.avg_pool_2(x)
        x_2 = self.conv_2(x_2)
        x_2 = ReLU()(x_2)
        x_2 = self.upsample_2(x_2)

        # Scale 3
        x_3 = self.avg_pool_3(x)
        x_3 = self.conv_3(x_3)
        x_3 = ReLU()(x_3)
        x_3 = self.upsample_3(x_3)

        # Scale 4
        x_4 = self.avg_pool_4(x)
        x_4 = self.conv_4(x_4)
        x_4 = ReLU()(x_4)
        x_4 = self.upsample_4(x_4)

        # Concatenate feature maps
        x = tf.concat((x, x_1, x_2, x_3, x_4), 3)

        # Up-sample fused features maps
        x = self.upsample_final(x)

        return x


class ResidualBlock(Model):
    """Standard ResNet block using instance normalization and ReLU applied to
    the fused feature maps."""
    def __init__(self, n_channels: int, init_gain: float, weight_decay: float):
        """Class constructor

        Attributes:
            n_channels: Number of input channels (equal to number of output
                        channels)
            init_gain: Initializer gain for orthogonal initialization
            weight_decay: Multiplicative factor for l2 weight regularization
        """
        super(ResidualBlock, self).__init__()
        self.conv_1 = Conv2D(filters=n_channels, kernel_size=(3, 3),
                padding='same', use_bias=False,
                kernel_initializer=orthogonal(gain=init_gain),
                kernel_regularizer=L1L2(l2=weight_decay))
        self.in_1 = LayerNormalization(axis=(1, 2), center=True, scale=True)
        self.relu = ReLU()
        self.conv_2 = Conv2D(filters=n_channels, kernel_size=(3, 3),
                padding='same', use_bias=True,
                kernel_initializer=orthogonal(gain=init_gain),
                kernel_regularizer=L1L2(l2=weight_decay))
        self.in_2 = LayerNormalization(axis=(1, 2), center=True, scale=True)

        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Perform call of Residual block

        Args:
            x: Input to the Residual block
        """
        # Store input for skip-connection
        identity = x

        # Residual pipeline

        x = self.in_1(x)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.in_2(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.in_2(x)

        # Skip-connection
        x += identity

        # Apply ReLU activation
        x = self.relu(x)

        return x


class ReflectionPadding2D(Layer):
    """Reflection padding layer."""
    def __init__(self, padding: Tuple[int, int]=(3, 3)):
        self.padding = padding
        super(ReflectionPadding2D, self).__init__()

        
 def call(self, x: tf.Tensor) -> tf.Tensor:
        """Perform call of Reflection Padding block

        Args:
            x: Input to the Reflection Padding block
        """
        w_pad, h_pad = self.padding

        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]],
                'REFLECT')


class SegmentationNetwork(Model):
    def __init__(self, n_classes: int, init_gain: float, weight_decay: float):
        """Class constructor

        Attributes:
            n_classes: The number of classes for segmentation (including
                       background)
            init_gain: Initializer gain for orthogonal initialization
            weight_decay: Weight decay constant
        """
        super(SegmentationNetwork, self).__init__()

        self.model_name = 'Segmentation_Network'

        self.n_classes = n_classes

        # First computational block (3 convolutional layers)
        self.ref_padding_1 = ReflectionPadding2D(padding=(3, 3))
        self.conv_block_1 = ConvolutionalBlock(filters=16, kernel_size=(7, 7),
                padding='valid', stride=1, init_gain=init_gain,
                weight_decay=weight_decay)
        self.conv_block_2 = ConvolutionalBlock(filters=32, kernel_size=(3, 3),
                padding='same', stride=2, init_gain=init_gain,
                weight_decay=weight_decay)
        self.conv_block_3 = ConvolutionalBlock(filters=64, kernel_size=(3, 3),
                padding='same', stride=2, init_gain=init_gain,
                weight_decay=weight_decay)
        self.block_1 = Sequential((self.ref_padding_1, self.conv_block_1,
            self.conv_block_2, self.conv_block_3))

        # Second computational block (3 residual blocks)
        self.res_block_1 = ResidualBlock(init_gain=init_gain, n_channels=64,
                weight_decay=weight_decay)
        self.res_block_2 = ResidualBlock(init_gain=init_gain, n_channels=64,
                weight_decay=weight_decay)
        self.res_block_3 = ResidualBlock(init_gain=init_gain, n_channels=64,
                weight_decay=weight_decay)
        self.block_2 = Sequential((self.res_block_1, self.res_block_2,
            self.res_block_3))

        # Third computational block (1 Pyramid Pooling Module)
        self.block_3 = PPM(init_gain=init_gain, input_shape=(32, 32, 64),
                weight_decay=weight_decay)

        # Fourth computational block (1 convolutional layer, 1 up-sampling
        # layer, 2 convolutional layers)
        self.conv_block_4 = ConvolutionalBlock(filters=34, kernel_size=(3, 3),
                padding='same', stride=1, init_gain=init_gain,
                weight_decay=weight_decay)
        self.upsample = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_block_5 = ConvolutionalBlock(filters=17, kernel_size=(3, 3),
                padding='same', stride=1, init_gain=init_gain,
                weight_decay=weight_decay)
        self.ref_padding_2 = ReflectionPadding2D(padding=(3, 3))

        if self.n_classes != 2:
            self.conv_final = Conv2D(filters=self.n_classes, kernel_size=(7, 7),
                    padding='valid', use_bias=True,
                    kernel_initializer=orthogonal(gain=init_gain),
                    kernel_regularizer=L1L2(l2=weight_decay))
        else:
            self.conv_final = Conv2D(filters=1, kernel_size=(7, 7),
                    padding='valid', use_bias=True,
                    kernel_initializer=orthogonal(gain=init_gain),
                    kernel_regularizer=L1L2(l2=weight_decay))

        self.block_4 = Sequential((self.conv_block_4, self.upsample,
            self.conv_block_5, self.ref_padding_2, self.conv_final))


    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Perform call of the segmentation network

        Args:
            x: Input to the segmentation block
        """
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        if self.n_classes == 2:
            x = tf.math.sigmoid(x)
            x = tf.concat((x, 1-x), axis=3)
        else:
            x = Softmax(axis=3)(x)

        return x


if __name__ == '__main__':
    # Prepare exemplary image batch
    input_path_1 = 'Datasets/Flowers/images/image_00023.jpg'
    input_path_2 = 'Datasets/Flowers/images/image_00081.jpg'
    image_1 = tf.image.decode_jpeg(tf.io.read_file(input_path_1))
    image_1 = tf.image.resize(image_1, (128, 128), preserve_aspect_ratio=False)
    image_1 = tf.expand_dims(tf.image.per_image_standardization(image_1), 0)
    image_2 = tf.image.decode_jpeg(tf.io.read_file(input_path_2))
    image_2 = tf.image.resize(tf.image.per_image_standardization(image_2),
            (128, 128), preserve_aspect_ratio=False)
    image_2 = tf.expand_dims(image_2, 0)
    image_batch = tf.concat((image_1, image_2), 0)

    # Create network object
    f = SegmentationNetwork(n_classes=2, init_gain=0.8, weight_decay=1e-4)

    # Forward pass
    mask_batch = f(image_batch)
    print('Output Shape: ', mask_batch.shape)

    fig, ax = plt.subplots(2, 2)
    image_1 = image_1[0].numpy()
    image_1 -= np.min(image_1)
    image_1 /= (np.max(image_1) - np.min(image_1))
    image_2 = image_2[0].numpy()
    
    image_2 -= np.min(image_2)
    image_2 /= (np.max(image_2) - np.min(image_2))
    mask_1 = Softmax(axis=2)(mask_batch[0]).numpy()[:, :, 1]
    mask_2 = Softmax(axis=2)(mask_batch[1]).numpy()[:, :, 1]

    # Plot images and masks
    ax[0, 0].imshow(image_1)
    ax[1, 0].imshow(image_2)
    ax[0, 1].imshow(mask_1, cmap='gray')
    ax[1, 1].imshow(mask_2, cmap='gray')
    plt.show()

