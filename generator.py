#!/usr/bin/python3

"""generator.py - Generator model for single classes and for k classes as
                  well as network components for these

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chiemelewski-Anders, Mats Steinweg & Bas Straathof"


import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, ReLU, \
        Conv2D, UpSampling2D, AveragePooling2D, Softmax
from tensorflow.keras.initializers import orthogonal
from tensorflow import random_uniform_initializer
from typing import Union, List, Tuple
from normalizations import InstanceNormalization
from network_components import SelfAttentionModule, SpectralNormalization
from train_utils import normalize_contrast
import matplotlib.pyplot as plt


class ConditionalBatchNormalization(Layer):
    """Use Conv2D layers to learn a function for gamma and beta instead of
    directly learning the shift and scale parameters of the BN layer."""
    def __init__(self, filters: int, init_gain: float):
        """Class constructor

        Attributes:
            filters: Number of filters in the convolutional layer
            init_gain: Initializer gain for orthogonal initialization
        """
        super(ConditionalBatchNormalization, self).__init__()

        # bias initialization constant (see PyTorch Conv2d documentation)
        self.k = tf.math.sqrt(1 / filters)

        # Instance Normalization | shifting and scaling switched off
        self.norm = InstanceNormalization(center=False, scale=False)

        # Learnable functions for mapping of noise vector to scale and shift
        # parameters gamma and beta
        self.gamma = Conv2D(filters=filters, kernel_size=(1, 1), use_bias=True, padding='same',
                            kernel_initializer=orthogonal(gain=init_gain),
                            bias_initializer=random_uniform_initializer(-self.k, self.k))
        self.beta = Conv2D(filters=filters, kernel_size=(1, 1), use_bias=True, padding='same',
                           kernel_initializer=orthogonal(gain=init_gain),
                           bias_initializer=random_uniform_initializer(-self.k, self.k))

    def call(self, x: tf.Tensor, z_k: tf.Tensor) -> tf.Tensor:
        """To call the CBN layer
        Args:
            x: Input tensor
            z_k: Noise vector for class k
        Returns:
            x: Output tensor of conditional batch normalization layer
        """
        # Pass input through Instance Normalization layer
        #x = self.norm(x)
        mean = tf.expand_dims(tf.math.reduce_mean(x, axis=(1, 2)), axis=1)
        mean = tf.expand_dims(mean, axis=2)
        std = tf.expand_dims(tf.math.reduce_std(x, axis=(1, 2)), axis=1)
        std = tf.expand_dims(std, axis=2)

        x = (x - mean) / std

        print(tf.math.reduce_mean(x[0], axis=(0, 1)))
        print(tf.math.reduce_std(x[0], axis=(0, 1)))
        # Get conditional gamma and beta
        gamma_c = self.gamma(z_k)
        beta_c = self.beta(z_k)

        # Compute output
        x = gamma_c * x + beta_c

        return x


class InputBlock(Layer):
    """First computational block of the generator network. Includes a
    fully-connected layer whose output is then reshaped to be able to start
    applying convolutional layers."""
    def __init__(self, init_gain: float, base_channels: int, output_factor: int, n_input: int):
        """Class constructor

        Attributes:
            init_gain: Initializer gain for orthogonal initialization
            base_channels: The number of base channels
            output_factor: Factor to reshape the output
        """
        super(InputBlock, self).__init__()

        # Number of output channels
        self.output_channels = base_channels*output_factor

        # Dimension of noise vector
        self.n_input = n_input

        # bias initialization constant (see PyTorch Conv2d documentation)
        self.k = tf.math.sqrt(1 / self.n_input)

        # Initial dense layer (implemented as 1x1 convolution) | Number of output channels*16
        # for reshaping into 4x4 feature maps
        self.dense = SpectralNormalization(Conv2D(filters=self.output_channels * 4 * 4, kernel_size=(1, 1),
                                                  kernel_initializer=orthogonal(gain=init_gain),
                                                  bias_initializer=random_uniform_initializer(-self.k, self.k)))

    def call(self, z_k: tf.Tensor, training: bool) -> tf.Tensor:
        """To call the first input block of the generator network

        Args:
            z_k: Noise vector for class k
            training: True if training phase

        Returns:
            x: Output tensor
        """
        # Reshape output of dense layer
        x = self.dense.call(z_k, training)
        x = tf.reshape(x, (-1, 4, 4, self.output_channels))

        return x


class ResidualUpsamplingBlock(Layer):
    def __init__(self, init_gain: float, base_channels: int, input_factor: int, output_factor: int, mask_scale: int):
        """Class constructor
        Args:
            init_gain: Initializer gain for orthogonal initialization
            base_channels: The number of base channels
            input_factor: Factor to reshape the input
            output_factor: Factor by which to multiply base_channels to get
                             final number of feature maps
            mask_scale: Down-scaling factor of segmentation mask to be
                        concatenated to the feature maps
        """
        super(ResidualUpsamplingBlock, self).__init__()

        # Number of input and output channels
        self.output_channels = base_channels*output_factor
        self.input_channels = base_channels*input_factor

        # bias initialization constants (see PyTorch Conv2d documentation)
        self.k_1 = tf.math.sqrt(1 / self.input_channels)
        self.k_2 = tf.math.sqrt(1 / (self.input_channels * 3 * 3))

        # Up-sampling layer
        self.upsample = UpSampling2D(size=(2, 2), interpolation='nearest')

        # Perform 1x1 convolutions on the identity to adjust the number of
        # channels to the output of the residual pipeline
        self.process_identity = tf.keras.Sequential()
        self.process_identity.add(UpSampling2D(size=(2, 2), interpolation='nearest'))
        self.process_identity.add(SpectralNormalization(Conv2D(filters=self.output_channels, kernel_size=(1, 1),
                                                               padding='same',
                                                               kernel_initializer=orthogonal(gain=init_gain),
                                                               bias_initializer=random_uniform_initializer(-self.k_1,
                                                                                                           self.k_1))))

        # Apply average-pooling to down-sample to segmentation mask
        self.mask_pool = AveragePooling2D(pool_size=mask_scale, padding='same')

        # Computational pipeline
        self.cbn_1 = ConditionalBatchNormalization(filters=self.input_channels, init_gain=init_gain)
        self.relu = ReLU()
        self.conv_1 = SpectralNormalization(Conv2D(filters=self.output_channels, kernel_size=(3, 3), padding='same',
                                                   kernel_initializer=orthogonal(gain=init_gain), use_bias=False))
        self.cbn_2 = ConditionalBatchNormalization(filters=self.output_channels, init_gain=init_gain)
        self.conv_2 = SpectralNormalization(Conv2D(filters=self.output_channels, kernel_size=(3, 3), padding='same',
                                                   kernel_initializer=orthogonal(gain=init_gain),
                                                   bias_initializer=random_uniform_initializer(-self.k_2, self.k_2)))

    def call(self, x: tf.Tensor, z_k: tf.Tensor, masks_k: tf.Tensor, training: bool) -> tf.Tensor:
        """To call the residual upsampling block
        Args:
            x: Input tensor
            z_k: Noise vector for class k
            masks_k: The masks
            training: Whether we are training
        Returns:
            x: Output tensor
        """

        # Residual pipeline
        # CBN and ReLU
        h = self.cbn_1.call(x, z_k)
        h = self.relu(h)

        # Down-sample and concatenate mask
        masks = self.mask_pool(masks_k)
        h = tf.concat((h, masks), axis=3)

        # Upsample feature maps
        h = self.upsample(h)
        h = self.conv_1.call(h, training)

        # CBN, ReLU, Conv2D
        h = self.cbn_2.call(h, z_k)
        h = self.relu(h)
        h = self.conv_2.call(h, training)

        # Process identity
        x = self.process_identity(x)

        # Skip-connection
        x += h

        return x


class OutputBlock(Layer):
    """Final block of the generator network:
        - Conditional Batch Norm
        - ReLU
        - Concatenate the mask
        - Conv 3x3
        - Tanh activation
    """
    def __init__(self, init_gain: float, base_channels: int, output_factor: int):
        """Class constructor

        Args:
            init_gain: Initializer gain for orthogonal initialization
            base_channels: The number of base channels
            output_factor: Factor to reshape the input
        """
        super(OutputBlock, self).__init__()

        # Number of output channels
        self.output_channels = base_channels*output_factor

        # bias initialization constants (see PyTorch Conv2d documentation)
        self.k = tf.math.sqrt(1 / (self.output_channels * 3 * 3))

        self.cbn = ConditionalBatchNormalization(filters=self.output_channels, init_gain=init_gain)
        self.relu = ReLU()
        self.conv = SpectralNormalization(Conv2D(filters=3, kernel_size=(3, 3), padding='same',
                                                 kernel_initializer=orthogonal(gain=init_gain)))

    def call(self, x: tf.Tensor, z_k: tf.Tensor, masks_k: tf.Tensor, training: bool) -> tf.Tensor:
        """To call the residual upsampling block
        Args:
            x: Input tensor
            z_k: Noise vector for class k
            masks_k: The masks
            training: Whether we are training
        Returns:
            x: Output tensor
        """

        # CBN and relu
        x = self.cbn.call(x, z_k)
        x = self.relu(x)

        # Concatenate masks
        x = tf.concat((x, masks_k), axis=3)

        # Perform convolution
        x = self.conv.call(x, training)

        # Tanh activation
        x = tf.keras.activations.tanh(x)

        return x


class ClassGenerator(Model):
    """Generator for region/class k"""
    def __init__(self, init_gain: float, k: int, base_channels: int = 32, n_input: int = 32):
        """Class constructor

        Attributes:
            init_gain: Initializer gain for orthogonal initialization
            k: Region for which the generator is trained
            base_channels: Data-dependent constant used for number of channels
                           throughout the network
            n_input: dimension of noise vector
        """
        super(ClassGenerator, self).__init__()

        # Class ID for generator
        self.k = k

        # Input dimension of noise vector
        self.n_input = n_input

        # Dataset-dependent constant for number of channels in network
        self.base_channels = base_channels

        # Dense layer + reshaping of noise vector to allow for convolutions
        self.block_1 = InputBlock(init_gain=init_gain, base_channels=self.base_channels, output_factor=16, n_input=32)

        # Residual Upsampling Blocks
        self.up_res_block_1 = ResidualUpsamplingBlock(init_gain=init_gain, base_channels=self.base_channels,
                                                      output_factor=16, input_factor=16, mask_scale=32)
        self.up_res_block_2 = ResidualUpsamplingBlock(init_gain=init_gain, base_channels=self.base_channels,
                                                      output_factor=8, input_factor=16, mask_scale=16)
        self.up_res_block_3 = ResidualUpsamplingBlock(init_gain=init_gain, base_channels=self.base_channels,
                                                      output_factor=4, input_factor=8, mask_scale=8)
        self.up_res_block_4 = ResidualUpsamplingBlock(init_gain=init_gain, base_channels=self.base_channels,
                                                      output_factor=2, input_factor=4, mask_scale=4)

        # Self-Attention Module
        self.block_3 = SelfAttentionModule(init_gain=init_gain, output_channels=2*base_channels)

        # Final residual up-sampling block
        self.block_4 = ResidualUpsamplingBlock(init_gain=init_gain, base_channels=self.base_channels, output_factor=1,
                                               input_factor=2, mask_scale=2)

        # Output block
        self.block_5 = OutputBlock(init_gain=init_gain, base_channels=self.base_channels, output_factor=1)

    def draw_region(self, z_k: tf.Tensor, masks_k: tf.Tensor, training: bool):
        """
        Draw fake region
        :param z_k: noise vector | shape: [batch_size, 1, 1, noise_dim]
        :param masks_k: masks | shape: [batch_size, 128, 128, 1]
        :param training: bool
        :return: fake region | shape: [batch_size, 128, 128, 3]
        """

        # First block | Output: [batch_size, 4, 4, 16*base_channels]
        x = self.block_1.call(z_k, training=training)

        # Second block | Residual Upsampling | Output: [batch_size, 64, 64, 2*base_channels]
        x = self.up_res_block_1.call(x, z_k, masks_k, training=training)

        x = self.up_res_block_2.call(x, z_k, masks_k, training=training)

        x = self.up_res_block_3.call(x, z_k, masks_k, training=training)

        x = self.up_res_block_4.call(x, z_k, masks_k, training=training)

        # Third block | Self Attention | Output: [batch_size, 64, 64, 2*base_channels]
        x = self.block_3.call(x, training=training)

        # Fourth Block | Residual Upsampling | Output: [batch_size, 128, 128, 1*base_channels]
        x = self.block_4.call(x, z_k, masks_k, training=training)

        # Fifth Block | Tanh Activation | Output: [batch_size, 128, 128, 3]
        x = self.block_5.call(x, z_k, masks_k, training=training)

        # Output | Multiply generated image with mask | Output: [batch_size, 128, 128, 3]
        region_k_fake = masks_k * x

        return region_k_fake

    def call(self, images_real: tf.Tensor, masks: tf.Tensor, z_k: tf.Tensor, n_regions: int,
             training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward pass of the class generator network - create a batch of fake images with respective class redrawn
        Args:
            images_real: Batch of images taken from the dataset
            masks: Extracted segmentation masks of shape:
                         [batch_size, height, width, n_classes-1]
            z_k: Noise vector for respective class
            n_regions: number of regions in the fake image
            training: Whether we are training
        Returns:
            batch_images_fake: Generated fake images
            batch_region_k_fake: Generated fake regions
        """

        images_k_fake = None

        # Get masks for region of this class generator
        masks_k = tf.expand_dims(masks[:, :, :, self.k], axis=3)

        for r in range(n_regions):

            # Redraw class of class generator
            if r == self.k:

                # get fake region
                regions_k_fake = self.draw_region(z_k, masks_k, training)

                # Add redrawn region to fake images
                if images_k_fake is None:
                    images_k_fake = regions_k_fake
                else:
                    images_k_fake += regions_k_fake

            # Reuse input image for other regions
            else:
                # Get inverted masks
                if n_regions == 2:
                    masks_r = 1.0 - masks_k
                else:
                    masks_r = tf.expand_dims(masks[:, :, :, r], axis=3)

                # add masked region of real image to fake image
                if images_k_fake is None:
                    images_k_fake = (images_real * masks_r)
                else:
                    images_k_fake += (images_real * masks_r)

        return images_k_fake, regions_k_fake


class Generator(Model):
    """Generator object containing a separate network for each region/class"""
    def __init__(self, n_classes: int, n_input: int, init_gain: float, base_channels: int):
        """Class constructor
        Attributes:
            n_classes: Number of regions to be generated, corresponding to the
                       number of classes in dataset.
            n_input: Dimensionality of the sampled input vector z_k
            init_gain: Gain for orthogonal initialization of network weights
            base_channels: Dataset-dependent constant for number of channels
        """
        super(Generator, self).__init__()

        # Set name for model saving
        self.model_name = 'Generator'

        # Number of classes modeled by generator
        self.n_classes = n_classes

        # Dimensionality of sampled noise vector
        self.n_input = n_input

        # List of class generator networks
        self.class_generators = [ClassGenerator(init_gain=init_gain, k=k,
                                                base_channels=base_channels) for k in range(self.n_classes)]

    def call(self, images_real: tf.Tensor, masks: tf.Tensor, z: tf.Tensor, training: bool) \
            -> Union[List[tf.Tensor], tf.Tensor]:
        """Generate fake images by separately redrawing each class using the
        segmentation masks for each image in the batch

        Args:
            images_real: Batch of training images of shape:
                               [batch_size, 128, 128, 3]
            masks: Predictions of segmentation network of shape:
                         [batch_size, 128, 128, n_classes]
            z: Noise vector | shape: [batch_size, n_classes, 1, 1, 32]
            training: Whether we are training
        Returns:
            if update_generator:
                batch_images_fake: Batch of fake images redrawn for each class
                                   of shape: [batch_size*n_classes, 128, 128, 3]
                batch_regions_fake: Batch of fake regions
                batch_z_k: Batch of
            else:
                batch_images_fake: Batch of fake images redrawn for each class
                                   of shape: [batch_size*n_classes, 128, 128, 3]
        """

        # Containers for fake images and fake regions
        images_fake, regions_fake = None, None

        for k in range(self.n_classes):

            # Generate batch of fake images
            images_k_fake, regions_k_fake = self.class_generators[k].call(images_real, masks, z[:, k],
                                                                          training=training, n_regions=self.n_classes)

            # Fake images from different regions are concatenated, fake regions are summed
            if images_fake is None:
                images_fake = images_k_fake
                regions_fake = regions_k_fake
            else:
                images_fake = tf.concat((images_fake, images_k_fake), axis=0)
                regions_fake += regions_k_fake

        return images_fake, regions_fake


