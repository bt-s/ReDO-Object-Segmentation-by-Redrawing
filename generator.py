import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, ReLU, Conv2D, UpSampling2D, AveragePooling2D, Softmax
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import orthogonal
from network_components import SpectralNormalization, SelfAttentionModule


###################################
# Conditional Batch Normalization #
###################################

class ConditionalBatchNormalization(Layer):
    """
    Conditional Batch Normalization Layer. Use a two-layer MLP to learn a function for gamma and beta instead of
    directly learning the shift and scale parameters of the BN layer.
    """
    def __init__(self, filters, init_gain):
        super(ConditionalBatchNormalization, self).__init__()

        # Instance Normalization | shifting and scaling switched off
        self.in_1 = LayerNormalization(axis=(1, 2), center=False, scale=False)

        # learnable functions for mapping of noise vector to scale and shift parameters gamma and beta
        self.gamma = Conv2D(filters=filters, kernel_size=(1, 1),
                            padding='same', kernel_initializer=orthogonal(gain=init_gain))
        self.beta = Conv2D(filters=filters, kernel_size=(1, 1),
                            padding='same', kernel_initializer=orthogonal(gain=init_gain))

    @tf.function
    def call(self, x, z_k):

        # pass input through Instance Normalization layer
        x = self.in_1(x)

        # get conditional gamma and beta
        gamma_c = self.gamma(z_k)
        beta_c = self.beta(z_k)

        # compute output
        x = gamma_c * x + beta_c

        return x


#################
# Initial Block #
#################

class InputBlock(Layer):
    """
    First computational block of the generator network. Includes a fully-connected layer whose output is then reshaped
    to be able to start applying convolutional layers. CBN and ReLU are also included.
    """
    def __init__(self, init_gain, base_channels, output_factor):
        super(InputBlock, self).__init__()

        # number of output channels
        self.output_channels = base_channels*output_factor

        # fully-connected layer with number of output channels * 4 * 4 units for reshaping into 4x4 feature maps
        self.dense = Dense(units=self.output_channels * 4 * 4, kernel_initializer=orthogonal(gain=init_gain))
        self.cbn = ConditionalBatchNormalization(filters=self.output_channels, init_gain=init_gain)
        self.relu = ReLU()

    @tf.function
    def call(self, z_k):

        # reshape output of fully-connected layer
        x = self.dense(z_k)
        x = tf.reshape(x, (-1, 4, 4, self.output_channels))

        # apply CBN
        x = self.cbn(x, z_k)
        x = self.relu(x)
        return x


##############################
# Residual Up-sampling Block #
##############################

class ResidualUpsamplingBlock(Layer):
    def __init__(self, init_gain, base_channels, input_factor, output_factor, mask_scale):
        super(ResidualUpsamplingBlock, self).__init__()

        # number of input and output channels
        self.output_channels = base_channels*output_factor
        self.input_channels = base_channels*input_factor

        # up-sampling layer
        self.upsample = UpSampling2D(size=(2, 2), interpolation='bilinear')

        # perform 1x1 convolutions on the identity to adjust the number of channels to the output of the computational
        # pipeline
        self.process_identity = Sequential()
        self.process_identity.add(self.upsample)
        self.process_identity.add(SpectralNormalization(Conv2D(filters=self.output_channels, kernel_size=(1, 1),
                                                    padding='same', kernel_initializer=orthogonal(gain=init_gain))))

        # apply average-pooling to down-sample to segmentation mask
        self.mask_pool = AveragePooling2D(pool_size=mask_scale, padding='same')

        # computational pipeline
        self.cbn_1 = ConditionalBatchNormalization(filters=self.input_channels, init_gain=init_gain)
        self.relu = ReLU()
        self.conv_1 = SpectralNormalization(Conv2D(filters=self.output_channels, kernel_size=(3, 3), padding='same',
                                       kernel_initializer=orthogonal(gain=init_gain)))
        self.cbn_2 = ConditionalBatchNormalization(filters=self.output_channels, init_gain=init_gain)
        self.conv_2 = SpectralNormalization(Conv2D(filters=self.output_channels, kernel_size=(3, 3), padding='same',
                                       kernel_initializer=orthogonal(gain=init_gain)))

    @tf.function
    def call(self, x, z_k, masks, training):

        # process identity
        identity = self.process_identity(x)

        # pass input through residual pipeline
        x = self.cbn_1(x, z_k)
        x = self.relu(x)
        # concatenate feature maps and masks
        masks = tf.cast(self.mask_pool(masks), tf.float32)  # resize masks to fit input shape
        x = tf.concat((x, masks), axis=3)
        x = self.upsample(x)
        x = self.conv_1(x, training)
        x = self.cbn_2(x, z_k)
        x = self.relu(x)
        x = self.conv_2(x, training)

        # skip-connection
        x += identity

        return x


###############
# Final Block #
###############

class OutputBlock(Layer):
    def __init__(self, init_gain, base_channels, output_factor):
        super(OutputBlock, self).__init__()

        # number of output channels
        self.output_channels = base_channels*output_factor

        self.cbn = ConditionalBatchNormalization(filters=self.output_channels, init_gain=init_gain)
        self.relu = ReLU()
        self.conv = SpectralNormalization(Conv2D(filters=3, kernel_size=(3, 3), padding='same',
                                                 kernel_initializer=orthogonal(gain=init_gain)))

    @tf.function
    def call(self, x, z_k, masks, training):
        x = self.cbn(x, z_k)
        x = self.relu(x)
        # concatenate feature maps and masks
        x = tf.concat((x, tf.cast(masks, tf.float32)), axis=3)
        x = self.conv(x, training)
        x = tf.keras.activations.tanh(x)

        return x


#####################
# Generator Network #
#####################

class Generator(Model):
    def __init__(self, init_gain, base_channels=32):
        super(Generator, self).__init__()

        # set name for model saving
        self.model_name = 'Generator'

        self.k = None  # region for which the generator is trained
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

    def call(self, batch_images_real, batch_masks_logits, noise_dim, training):
        """
        Forward pass of the generator network. Create a batch of fake images.
        :param batch_images_real: batch of images taken from the dataset
        :param batch_masks_logits: extracted segmentation masks | shape: [batch_size, height, width, n_classes-1]
        :param noise_dim: dimensionality of the noise vector
        :param training: current network phase to switch between modes for CBN layers
        :return: batch of fake images | shape: [batch_size, height, width, 3]
        """

        # batch size
        batch_size = batch_masks_logits.shape[0]

        # number of different regions
        n_regions = batch_masks_logits.shape[3]

        # sample noise vector
        z_k = tf.random.normal([batch_size, 1, 1, noise_dim])

        # re-draw image
        batch_images_fake = tf.zeros(batch_images_real.shape)
        for k in range(n_regions):

            # get region mask
            batch_masks_k = tf.expand_dims(Softmax(axis=3)(batch_masks_logits)[:, :, :, k], axis=3)

            # re-draw sampled region
            if k == self.k:
                x = self.block_1(z_k, training=training)
                x = self.up_res_block_1(x, z_k, batch_masks_k, training=training)
                x = self.up_res_block_2(x, z_k, batch_masks_k, training=training)
                x = self.up_res_block_3(x, z_k, batch_masks_k, training=training)
                x = self.up_res_block_4(x, z_k, batch_masks_k, training=training)
                x = self.block_3(x, training=training)
                x = self.block_4(x, z_k, batch_masks_k, training=training)
                batch_regions_fake = self.block_5(x, z_k, batch_masks_k, training=training)

                # add redrawn regions to batch of fake images
                batch_images_fake += batch_regions_fake * batch_masks_k

            # re-use input image for other regions
            else:
                batch_images_fake += batch_images_real * batch_masks_k

        return batch_images_fake, z_k[:, 0, 0, :]

    def set_region(self, k):
        self.k = k

    def set_name(self, name):
        self.model_name = name


if __name__ == '__main__':

    generator = Generator(init_gain=1.0)
    generator.set_region(0)
    optimizer = Adam(learning_rate=1e-1, beta_1=0, beta_2=0.9)
    input_path_1 = 'Datasets/Birds/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
    label_path_1 = 'Datasets/Birds/labels/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.png'
    image_real = tf.image.decode_jpeg(tf.io.read_file(input_path_1))
    image_real = tf.image.resize(image_real, (128, 128), preserve_aspect_ratio=False)
    image_real = tf.expand_dims(tf.image.per_image_standardization(image_real), 0)
    mask = tf.image.decode_png(tf.io.read_file(label_path_1))
    mask = tf.expand_dims(tf.image.resize(mask, (128, 128), preserve_aspect_ratio=False), 0)
    mask = (mask / 255.0 * 2) - 1.0
    mask = tf.concat((mask, -1*mask), axis=3)

    with tf.GradientTape() as tape:
        image_fake, z_k = generator(image_real, mask, noise_dim=32, training=True)

    gradients = tape.gradient(image_fake, generator.trainable_variables)
    # update weights
    optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

    # plot output
    fig, ax = plt.subplots(1, 3)
    image_fake = image_fake[0].numpy()
    image_fake -= np.min(image_fake)
    image_fake /= (np.max(image_fake) - np.min(image_fake))
    image_real = image_real[0].numpy()
    image_real -= np.min(image_real)
    image_real /= (np.max(image_real) - np.min(image_real))
    ax[0].imshow(image_real)
    ax[1].imshow(mask[0].numpy()[:, :, 1], cmap='gray')
    ax[2].imshow(image_fake)
    plt.show()
