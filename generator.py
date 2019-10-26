#!/usr/bin/python3

"""generator.py - Generator model for single classes and for k classes as
                  well as network components for these

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chiemelewski-Anders, Mats Steinweg & Bas Straathof"


import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, ReLU, \
        Conv2D, UpSampling2D, AveragePooling2D, Softmax
from tensorflow.keras.initializers import orthogonal
from typing import Union, List, Tuple
from normalizations import InstanceNormalization
from network_components import SelfAttentionModule, SpectralNormalization


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

        # Instance Normalization | shifting and scaling switched off
        self.in_1 = InstanceNormalization(center=False, scale=False)

        # Learnable functions for mapping of noise vector to scale and shift
        # parameters gamma and beta
        self.gamma = Conv2D(filters=filters, kernel_size=(1, 1), use_bias=True,
                padding='same', kernel_initializer=orthogonal(gain=init_gain))
        self.beta = Conv2D(filters=filters, kernel_size=(1, 1), use_bias=True,
                padding='same', kernel_initializer=orthogonal(gain=init_gain))

    def call(self, x: tf.Tensor, z_k: tf.Tensor) -> tf.Tensor:
        """To call the CBN layer

        Args:
            x: Input tensor
            z_k: Noise vector for class k

        Returns:
            x: Output tensor of conditional batch normalization layer
        """
        # Pass input through Instance Normalization layer
        x = self.in_1(x)

        # Get conditional gamma and beta
        gamma_c = self.gamma(z_k)
        beta_c = self.beta(z_k)

        # Compute output
        x = gamma_c * x + beta_c

        return x


class InputBlock(Layer):
    """First computational block of the generator network. Includes a
    fully-connected layer whose output is then reshaped to be able to start
    applying convolutional layers. CBN and ReLU are also included."""
    def __init__(self, init_gain: float, base_channels: int,
            output_factor: int):
        """Class constructor

        Attributes:
            init_gain: Initializer gain for orthogonal initialization
            base_channels: The number of base channels
            output_factor: Factor to reshape the output
        """
        super(InputBlock, self).__init__()

        # Number of output channels
        self.output_channels = base_channels*output_factor

        # Fully-connected layer with number of output channels * 4 * 4 units
        # for reshaping into 4x4 feature maps
        self.dense = Dense(units=self.output_channels * 4 * 4,
                kernel_initializer=orthogonal(gain=init_gain))
        self.cbn = ConditionalBatchNormalization(filters=self.output_channels,
                init_gain=init_gain)
        self.relu = ReLU()

    def call(self, z_k: tf.Tensor) -> tf.Tensor:
        """To call the first input block of the generator network

        Args:
            z_k: Noise vector for class k

        Returns:
            x: Output tensor
        """
        # Reshape output of fully-connected layer
        x = self.dense(z_k[:, 0, 0, :])
        x = tf.reshape(x, (-1, 4, 4, self.output_channels))

        # Apply CBN
        x = self.cbn(x, z_k)
        x = self.relu(x)

        return x


class ResidualUpsamplingBlock(Layer):
    def __init__(self, init_gain: float, base_channels: int, input_factor: int,
            output_factor: int, mask_scale: int):
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

        # Up-sampling layer
        self.upsample = UpSampling2D(size=(2, 2), interpolation='bilinear')

        # Perform 1x1 convolutions on the identity to adjust the number of
        # channels to the output of the computational
        self.process_identity = tf.keras.Sequential()
        self.process_identity.add(UpSampling2D(size=(2, 2), interpolation='bilinear'))
        self.process_identity.add(SpectralNormalization(Conv2D(
            filters=self.output_channels, kernel_size=(1, 1), padding='same',
            kernel_initializer=orthogonal(gain=init_gain))))

        # Apply average-pooling to down-sample to segmentation mask
        self.mask_pool = AveragePooling2D(pool_size=mask_scale, padding='same')

        # Computational pipeline
        self.cbn_1 = ConditionalBatchNormalization(filters=self.input_channels,
                init_gain=init_gain)
        self.relu = ReLU()
        self.conv_1 = SpectralNormalization(Conv2D(
            filters=self.output_channels, kernel_size=(3, 3), padding='same',
            kernel_initializer=orthogonal(gain=init_gain)))
        self.cbn_2 = ConditionalBatchNormalization(filters=self.output_channels,
                init_gain=init_gain)
        self.conv_2 = SpectralNormalization(Conv2D(
            filters=self.output_channels, kernel_size=(3, 3), padding='same',
            kernel_initializer=orthogonal(gain=init_gain)))

    def call(self, x: tf.Tensor, z_k: tf.Tensor, masks: tf.Tensor,
            training: bool) -> tf.Tensor:
        """To call the residual upsampling block

        Args:
            x: Input tensor
            z_k: Noise vector for class k
            masks: The masks
            training: Whether we are training

        Returns:
            x: Output tensor
        """
        # Process identity
        identity = self.process_identity(x)

        # Res block computations
        x = self.cbn_1(x, z_k)
        x = self.relu(x)
        # Down-sample and concatenate mask
        masks = tf.cast(self.mask_pool(masks), tf.float32)
        x = tf.concat((x, masks), axis=3)
        # Upsample feature maps
        x = self.upsample(x)
        x = self.conv_1(x, training)
        x = self.cbn_2(x, z_k)
        x = self.relu(x)
        x = self.conv_2(x, training)

        # Skip-connection
        x += identity

        return x


class OutputBlock(Layer):
    """Final block of the generator network:
        - Conditional Batch Norm
        - ReLU
        - Concatenate the mask
        - Conv 3x3
        - Tanh activation
    """
    def __init__(self, init_gain: float, base_channels: int,
            output_factor: int):
        """Class constructor

        Args:
            init_gain: Initializer gain for orthogonal initialization
            base_channels: The number of base channels
            output_factor: Factor to reshape the input
        """
        super(OutputBlock, self).__init__()

        # Number of output channels
        self.output_channels = base_channels*output_factor

        self.cbn = ConditionalBatchNormalization(filters=self.output_channels,
                init_gain=init_gain)
        self.relu = ReLU()
        self.conv = SpectralNormalization(Conv2D(
            filters=3, kernel_size=(3, 3), padding='same',
            kernel_initializer=orthogonal(gain=init_gain)))

    def call(self, x: tf.Tensor, z_k: tf.Tensor, masks: tf.Tensor,
            training: bool) -> tf.Tensor:
        """To call the residual upsampling block

        Args:
            x: Input tensor
            z_k: Noise vector for class k
            masks: The masks
            training: Whether we are training

        Returns:
            x: Output tensor
        """
        x = self.cbn(x, z_k)
        x = self.relu(x)

        # Concatenate feature maps and masks
        x = tf.concat((x, tf.cast(masks, tf.float32)), axis=3)
        x = self.conv(x, training)
        x = tf.keras.activations.tanh(x)

        return x


class ClassGenerator(Model):
    """Generator for region/class k"""
    def __init__(self, init_gain: float, k: int, base_channels: int=32):
        """Class constructor

        Attributes:
            init_gain: Initializer gain for orthogonal initialization
            k: Region for which the generator is trained
            base_channels: Data-dependent constant used for number of channels
                           throughout the network
        """
        super(ClassGenerator, self).__init__()

        # Class ID for generator
        self.k = k

        # Dataset-dependent constant for number of channels in network
        self.base_channels = base_channels

        # Fully-connected layer + reshaping of noise vector to allow for
        # convolutions
        self.block_1 = InputBlock(init_gain=init_gain,
                base_channels=self.base_channels, output_factor=16)

        # Residual Upsampling Blocks
        self.up_res_block_1 = ResidualUpsamplingBlock(init_gain=init_gain,
                base_channels=self.base_channels, output_factor=16,
                input_factor=16, mask_scale=32)
        self.up_res_block_2 = ResidualUpsamplingBlock(init_gain=init_gain,
                base_channels=self.base_channels, output_factor=8,
                input_factor=16, mask_scale=16)
        self.up_res_block_3 = ResidualUpsamplingBlock(init_gain=init_gain,
                base_channels=self.base_channels, output_factor=4,
                input_factor=8, mask_scale=8)
        self.up_res_block_4 = ResidualUpsamplingBlock(init_gain=init_gain,
                base_channels=self.base_channels, output_factor=2,
                input_factor=4, mask_scale=4)

        # Self-Attention Module
        self.block_3 = SelfAttentionModule(init_gain=init_gain,
                output_channels=2*base_channels)

        # Final residual up-sampling block
        self.block_4 = ResidualUpsamplingBlock(init_gain=init_gain,
                base_channels=self.base_channels, output_factor=1,
                input_factor=2, mask_scale=2)

        # Output block
        self.block_5 = OutputBlock(init_gain=init_gain,
                base_channels=self.base_channels, output_factor=1)

    def call(self, batch_images_real: tf.Tensor, batch_masks: tf.Tensor, z_k: tf.Tensor,
            n_input: Tuple, training: bool) -> Tuple[tf.Tensor, tf.Tensor,
                    tf.Tensor]:
        """Forward pass of the generator network - create a batch of fake images

        Args:
            batch_images_real: Batch of images taken from the dataset
            batch_masks: Extracted segmentation masks of shape:
                         [batch_size, height, width, n_classes-1]
            n_input: Dimensionality of the noise vector
            training: Whether we are training

        Returns:
            batch_images_fake: Generated fake images
            batch_region_k_fake: Generated fake regions
            z_k: Generated noise vector
        """

        # Number of different regions
        n_regions = batch_masks.shape[3]

        # Get masks for region k
        batch_masks_k = tf.expand_dims(batch_masks[:, :, :, self.k], axis=3)

        # Container for re-drawn image
        batch_images_fake = tf.zeros(batch_images_real.shape)

        for k in range(n_regions):
            # Re-draw sampled region
            if k == self.k:
                x = self.block_1(z_k)
                x = self.up_res_block_1(x, z_k, batch_masks_k,
                        training=training)
                x = self.up_res_block_2(x, z_k, batch_masks_k,
                        training=training)
                x = self.up_res_block_3(x, z_k, batch_masks_k,
                        training=training)
                x = self.up_res_block_4(x, z_k, batch_masks_k,
                        training=training)
                x = self.block_3(x, training=training)
                x = self.block_4(x, z_k, batch_masks_k, training=training)
                batch_region_k_fake = self.block_5(x, z_k, batch_masks_k,
                        training=training)
                batch_region_k_fake *= batch_masks_k

                # Add redrawn regions to batch of fake images
                batch_images_fake += batch_region_k_fake

            # Re-use input image for other regions
            else:
                if n_regions == 2:
                    batch_masks_inv = 1.0 - batch_masks_k
                else:
                    batch_masks_inv = tf.expand_dims(batch_masks[:, :, :, k],
                            axis=3)

                batch_images_fake += batch_images_real * batch_masks_inv

        return batch_images_fake, batch_region_k_fake


class Generator(Model):
    """Generator object containing a separate network for each region/class"""
    def __init__(self, n_classes: int, n_input: int, init_gain: float,
            base_channels: int):
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

    def call(self, batch_images_real: tf.Tensor, batch_masks: tf.Tensor, z: tf.Tensor,
            update_generator: bool, training: bool) -> Union[List[tf.Tensor],
                tf.Tensor]:
        """Generate fake images by separately redrawing each class using the
        segmentation masks for each image in the batch

        Args:
            batch_images_real: Batch of training images of shape:
                               [batch_size, 128, 128, 3]
            batch_masks: Predictions of segmentation network of shape:
                         [batch_size, 128, 128, n_classes]
            update_generator: Whether the function is called during generator
                              update.
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
        batch_images_fake, batch_regions_fake, batch_z_k = None, None, None

        for k in range(self.n_classes):

            # get noise vector for class k
            z_k = z[:, k]

            # Generate batch of fake images
            batch_images_k_fake, batch_region_k_fake = \
                            self.class_generators[k](batch_images_real, batch_masks, z_k,
                                    n_input=self.n_input, training=training)

            if batch_images_fake is None:
                batch_images_fake = batch_images_k_fake
                batch_regions_fake = batch_region_k_fake
            else:
                batch_images_fake = tf.concat((batch_images_fake, batch_images_k_fake), axis=0)
                batch_regions_fake += batch_region_k_fake

        # Return batch of fake images
        if update_generator:
            return batch_images_fake, batch_regions_fake
        else:
            return batch_images_fake

