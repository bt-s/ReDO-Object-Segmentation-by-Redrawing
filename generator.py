#!/usr/bin/python3

"""generator.py - <explanation-here>.

For the NeurIPS Reproducibility Challange and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chiemelewski-Anders, Mats Steinweg & Bas Straathof"


import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, ReLU, \
        Conv2D, UpSampling2D, AveragePooling2D, Softmax
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import orthogonal

from network_components import SelfAttentionModule, SpectralNormalization
from discriminator import Discriminator
from information_network import InformationConservationNetwork
from train_utils import UnsupervisedLoss


class ConditionalBatchNormalization(Layer):
    """Conditional Batch Normalization Layer.
    Use Conv2D layers to learn a function for gamma and beta instead of directly
    learning the shift and scale parameters of the BN layer."""
    def __init__(self, filters, init_gain):
        """Class constructior

        Args:
            filters (?):
            init_gain (?):
        """
        super(ConditionalBatchNormalization, self).__init__()

        # Instance Normalization | shifting and scaling switched off
        self.in_1 = LayerNormalization(axis=(1, 2), center=False, scale=False)

        # Learnable functions for mapping of noise vector to scale and shift
        # parameters gamma and beta
        self.gamma = Conv2D(filters=filters, kernel_size=(1, 1), use_bias=True,
                padding='same', kernel_initializer=orthogonal(gain=init_gain))
        self.beta = Conv2D(filters=filters, kernel_size=(1, 1), use_bias=True,
                padding='same', kernel_initializer=orthogonal(gain=init_gain))


    def call(self, x, z_k):
        """To call the CBN layer

        Args:
            x (?):
            z_k (?):

        Returns:
            x (?):
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
    applying convolutional layers. CBN and ReLU are also included.
    """
    def __init__(self, init_gain, base_channels, output_factor):
        """Class constructior

        Args:
            init_gain (?):
            base_channels (?):
            output_factor (?):
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


    def call(self, z_k):
        """To call the input block layer

        Args:
            z_k (?):

        Returns:
            x (?):
        """

        # Reshape output of fully-connected layer
        x = self.dense(z_k)
        x = tf.reshape(x, (-1, 4, 4, self.output_channels))

        # Apply CBN
        x = self.cbn(x, z_k)
        x = self.relu(x)

        return x


class ResidualUpsamplingBlock(Layer):
    """Description here"""
    def __init__(self, init_gain, base_channels, input_factor, output_factor,
            mask_scale):
        """Class constructior

        Args:
            init_gain (?):
            base_channels (?):
            input_factor (?):
            output_factor (?):
            mask_scale (?):
        """
        super(ResidualUpsamplingBlock, self).__init__()

        # Number of input and output channels
        self.output_channels = base_channels*output_factor
        self.input_channels = base_channels*input_factor

        # Up-sampling layer
        self.upsample = UpSampling2D(size=(2, 2), interpolation='bilinear')

        # Perform 1x1 convolutions on the identity to adjust the number of
        # channels to the output of the computational
        self.process_identity = Sequential()
        self.process_identity.add(self.upsample)
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


    def call(self, x, z_k, masks, training):
        """To call the residiual upsampling block layer

        Args:
            z_k (?):
            masks (?):
            training (?):

        Returns:
            x (?):
        """
        # Process identity
        identity = self.process_identity(x)

        # Pass input through residual pipeline
        x = self.cbn_1(x, z_k)
        x = self.relu(x)

        # Concatenate feature maps and masks
        masks = tf.cast(self.mask_pool(masks), tf.float32)
        x = tf.concat((x, masks), axis=3)
        x = self.upsample(x)
        x = self.conv_1(x, training)
        x = self.cbn_2(x, z_k)
        x = self.relu(x)
        x = self.conv_2(x, training)

        # Skip-connection
        x += identity

        return x


class OutputBlock(Layer):
    """Description here"""
    def __init__(self, init_gain, base_channels, output_factor):
        """Class constructior

        Args:
            init_gain (?):
            base_channels (?):
            output_factor (?):
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


    def call(self, x, z_k, masks, training):
        """To call the residiual upsampling block layer

        Args:
            z_k (?):
            masks (?):
            training (?):

        Returns:
            x (?):
        """
        x = self.cbn(x, z_k)
        x = self.relu(x)

        # Concatenate feature maps and masks
        x = tf.concat((x, tf.cast(masks, tf.float32)), axis=3)
        x = self.conv(x, training)
        x = tf.keras.activations.tanh(x)

        return x


class ClassGenerator(Model):
    """Description here"""
    def __init__(self, init_gain, k, base_channels=32):
        super(ClassGenerator, self).__init__()

        self.k = k  # region for which the generator is trained
        self.base_channels = base_channels  # data-dependent constant used for number of channels throughout the network

        # first computational block | fully-connected layer + reshaping of noise vector to allow for convolutions
        self.block_1 = InputBlock(init_gain=init_gain, base_channels=self.base_channels, output_factor=16)

        # second computational block | residual up-sampling layers
        # mask_scale: down-scaling factor of segmentation mask to be concatenated to the feature maps
        # output_channels: factor by which to multiply base_channels to get final number of feature maps
        self.up_res_block_1 = ResidualUpsamplingBlock(init_gain=init_gain, base_channels=self.base_channels,
                                                      output_factor=16, input_factor=16, mask_scale=32)
        self.up_res_block_2 = ResidualUpsamplingBlock(init_gain=init_gain, base_channels=self.base_channels,
                                                      output_factor=8, input_factor=16, mask_scale=16)
        self.up_res_block_3 = ResidualUpsamplingBlock(init_gain=init_gain, base_channels=self.base_channels,
                                                      output_factor=4, input_factor=8, mask_scale=8)
        self.up_res_block_4 = ResidualUpsamplingBlock(init_gain=init_gain, base_channels=self.base_channels,
                                                      output_factor=2, input_factor=4, mask_scale=4)

        # computational block 3 | self-attention module
        self.block_3 = SelfAttentionModule(init_gain=init_gain, output_channels=2*base_channels)

        # computational block 4 | final residual up-sampling block
        self.block_4 = ResidualUpsamplingBlock(init_gain=init_gain, base_channels=self.base_channels,
                                                      output_factor=1, input_factor=2, mask_scale=2)
        # computational block 5 | output block
        self.block_5 = OutputBlock(init_gain=init_gain, base_channels=self.base_channels, output_factor=1)

    def call(self, batch_images_real, batch_masks, n_input, training):
        """
        Forward pass of the generator network. Create a batch of fake images.
        :param batch_images_real: batch of images taken from the dataset
        :param batch_masks: extracted segmentation masks | shape: [batch_size, height, width, n_classes-1]
        :param n_input: dimensionality of the noise vector
        :param training: current network phase to switch between modes for CBN layers
        :return: batch of fake images | shape: [batch_size, height, width, 3]
        """

        # batch size
        batch_size = batch_masks.shape[0]

        # number of different regions
        n_regions = batch_masks.shape[3]

        # sample noise vector
        z_k = tf.random.normal([batch_size, 1, 1, n_input])

        # get masks for region k
        batch_masks_k = tf.expand_dims(batch_masks[:, :, :, self.k], axis=3)

        # re-draw image
        batch_images_fake = tf.zeros(batch_images_real.shape)
        for k in range(n_regions):

            # re-draw sampled region
            if k == self.k:
                x = self.block_1(z_k)
                x = self.up_res_block_1(x, z_k, batch_masks_k, training=training)
                x = self.up_res_block_2(x, z_k, batch_masks_k, training=training)
                x = self.up_res_block_3(x, z_k, batch_masks_k, training=training)
                x = self.up_res_block_4(x, z_k, batch_masks_k, training=training)
                x = self.block_3(x, training=training)
                x = self.block_4(x, z_k, batch_masks_k, training=training)
                batch_region_k_fake = self.block_5(x, z_k, batch_masks_k, training=training)
                batch_region_k_fake = batch_region_k_fake * batch_masks_k

                # add redrawn regions to batch of fake images
                batch_images_fake += batch_region_k_fake

            # re-use input image for other regions
            else:
                if n_regions == 2:
                    batch_masks_inv = 1.0 - batch_masks_k
                else:
                    batch_masks_inv = tf.expand_dims(batch_masks[:, :, :, k], axis=3)

                batch_images_fake += batch_images_real * batch_masks_inv

        return batch_images_fake, batch_region_k_fake, z_k[:, 0, 0, :]


#####################
# Generator Network #
#####################

class Generator(Model):
    def __init__(self, n_classes, n_input, init_gain, base_channels):
        """
        Generator object that contains a separate network for each region.
        :param n_classes: number of regions to be generated. Corresponds to number of classes in dataset.
        :param n_input: dimensionality of the sampled input vector z_k
        :param init_gain: gain for orthogonal initialization of network weights
        :param base_channels: dataset-dependent constant for number of channels
        """
        super(Generator, self).__init__()

        # set name for model saving
        self.model_name = 'Generator'

        # number of classes modeled by generator
        self.n_classes = n_classes

        # dimensionality of sampled noise vector
        self.n_input = n_input

        # list of class generator networks
        self.class_generators = [ClassGenerator(init_gain=init_gain, k=k, base_channels=base_channels)
                                  for k in range(self.n_classes)]

    def call(self, batch_images_real, batch_masks, update_generator, training):
        """
        Generate fake images by separately redrawing each class using the segmentation masks for each image in the batch
        :param batch_images_real: batch of training images | shape: [batch_size, 128, 128, 3]
        :param batch_masks: predictions of segmentation network | shape: [batch_size, 128, 128, n_classes]
        :param update_generator: True if function called during generator update. Returns noise vector and estimated
        noise vector along with batch of fake images.
        :param training: True if function called during training.
        :return: batch of fake images redrawn for each class | shape: [batch_size*n_classes, 128, 128, 3]
        """

        batch_images_fake = None
        batch_z_k = None
        batch_regions_fake = None

        for k in range(self.n_classes):

            # get batch of fake images for respective region
            batch_images_k_fake, batch_region_k_fake, z_k = self.class_generators[k](batch_images_real, batch_masks,
                                                                n_input=self.n_input, training=training)

            if batch_images_fake is None:
                batch_images_fake = batch_images_k_fake
                batch_regions_fake = batch_region_k_fake
                if update_generator:
                    batch_z_k = z_k
            else:
                batch_images_fake = tf.concat((batch_images_fake, batch_images_k_fake), axis=0)
                if update_generator:
                    batch_z_k = tf.concat((batch_z_k, z_k), axis=0)
                    batch_regions_fake += batch_region_k_fake

        # return batch of fake images
        if update_generator:
            return batch_images_fake, batch_regions_fake, batch_z_k
        else:
            return batch_images_fake


if __name__ == '__main__':

    # create generator object
    generator = Generator(n_classes=2, n_input=32, base_channels=32, init_gain=1.0)

    # discriminator network
    discriminator = Discriminator(init_gain=1.0)

    # information network
    information_network = InformationConservationNetwork(init_gain=1.0, n_classes=2, n_output=32)

    # create optimizer object
    optimizer = Adam(learning_rate=1e-1, beta_1=0, beta_2=0.9)

    # create loss function
    loss = UnsupervisedLoss(lambda_z=5.0)

    # load input image and mask
    input_path_1 = 'Datasets/Flowers/images/image_00001.jpg'
    label_path_1 = 'Datasets/Flowers/labels/label_00001.jpg'
    image_real = tf.image.decode_jpeg(tf.io.read_file(input_path_1))
    image_real = tf.image.resize(image_real, (128, 128), preserve_aspect_ratio=False)
    image_real = tf.expand_dims(tf.image.per_image_standardization(image_real), 0)
    mask = tf.image.decode_jpeg(tf.io.read_file(label_path_1), channels=1)
    mask = tf.expand_dims(tf.image.resize(mask, (128, 128), preserve_aspect_ratio=False), 0)
    background_color = 29
    mask = tf.expand_dims(tf.cast(tf.where(tf.logical_or(mask <= 0.9 * background_color, mask >= 1.1 * background_color)
                                           , 10, -10), tf.float32)[:, :, :, 0], 3)
    masks = tf.concat((mask, -1*mask), axis=3)
    masks = tf.math.sigmoid(masks)

    with tf.GradientTape() as tape:
        batch_image_fake, batch_regions_fake, z_k = generator(image_real, masks, update_generator=True, training=True)
        z_k_hat = information_network(batch_regions_fake, training=True)
        d_logits_fake = discriminator(batch_image_fake, training=True)
        g_loss_d, g_loss_i = loss.get_g_loss(d_logits_fake, z_k, z_k_hat)
        g_loss = g_loss_d + g_loss_i
        print('G_D: ', g_loss_d)
        print('G_I: ', g_loss_i)

    gradients = tape.gradient(g_loss, generator.trainable_variables)
    # update weights
    optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

    # input image
    image_real = image_real[0].numpy()
    image_real -= np.min(image_real)
    image_real /= (np.max(image_real) - np.min(image_real))
    # fake image with redrawn foreground
    image_fake_fg = batch_image_fake[0].numpy()
    image_fake_fg -= np.min(image_fake_fg)
    image_fake_fg /= (np.max(image_fake_fg) - np.min(image_fake_fg))
    # fake image with redrawn background
    image_fake_bg = batch_image_fake[1].numpy()
    image_fake_bg -= np.min(image_fake_bg)
    image_fake_bg /= (np.max(image_fake_bg) - np.min(image_fake_bg))

    # plot output
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].set_title('Input Image')
    ax[0, 0].imshow(image_real)
    ax[0, 1].set_title('Mask Foreground')
    ax[0, 1].imshow(masks[0].numpy()[:, :, 1], cmap='gray')
    ax[1, 0].set_title('Fake Foreground')
    ax[1, 0].imshow(image_fake_fg)
    ax[1, 1].set_title('Fake Background')
    ax[1, 1].imshow(image_fake_bg)
    plt.show()
