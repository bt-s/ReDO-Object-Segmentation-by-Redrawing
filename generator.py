import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, ReLU, Conv2D, UpSampling2D, AveragePooling2D, Softmax
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
    def __init__(self, n_input, n_hidden, n_output):
        super(ConditionalBatchNormalization, self).__init__()

        self.n_input = n_input  # number of dimensions of the conditioning noise vector
        self.n_hidden = n_hidden  # number of hidden units in MLPs
        self.n_output = n_output  # number of output channels for MLPs corresponding to feature channels in
        # current layer

        self.beta = tf.Variable(0.0, name='cbn_beta')  # trainable scale variable
        self.gamma = tf.Variable(1.0, name='cbn_gamma')  # trainable shift variable

        self.bn = BatchNormalization(axis=3, center=False, scale=False)  # basic BN layer for normalization | shifting
        # and scaling switched off

        # Two-Layer MLPs used to predict betas and gammas
        self.GammaMLP = Sequential((
            Dense(units=n_hidden, input_shape=(n_input, )),
            ReLU(),
            Dense(units=n_output)))

        self.BetaMLP = Sequential((
            Dense(units=n_hidden, input_shape=(n_input, )),
            ReLU(),
            Dense(units=n_output)))

    @tf.function
    def call(self, x, noise, training):

        x = self.bn(x, training=training)  # pass input through normal BN layer

        # get parameter offsets from MLPs and reshape to match input dimensions
        delta_gamma = self.GammaMLP(noise)
        delta_gamma = tf.tile(tf.expand_dims(tf.expand_dims(delta_gamma, axis=1), axis=2),
                               [1, x.shape[1], x.shape[2], 1])
        delta_beta = self.BetaMLP(noise)
        delta_beta = tf.tile(tf.expand_dims(tf.expand_dims(delta_beta, axis=1), axis=2),
                              [1, x.shape[1], x.shape[2], 1])

        # compute conditional shift and scale parameters
        gamma_c = self.gamma + delta_gamma
        beta_c = self.beta + delta_beta

        # compute output
        output = gamma_c * x + beta_c

        return output


#################
# Initial Block #
#################

class InputBlock(Layer):
    """
    First computational block of the generator network. Includes a fully-connected layer whose output is then reshaped
    to be able to start applying convolutional layers. CBN and ReLU are also included.
    """
    def __init__(self, init_gain, input_dim, base_channels, output_channels):
        super(InputBlock, self).__init__()
        self.base_channels = base_channels
        self.fc = Dense(units=base_channels * 16 * 4 * 4, kernel_initializer=orthogonal(gain=init_gain))
        self.cbn = ConditionalBatchNormalization(input_dim, 256, base_channels*output_channels)
        self.relu = ReLU()

    @tf.function
    def call(self, z_k, training):

        # reshape output of fully-connected layer
        x = self.fc(z_k)
        x = tf.reshape(x, (-1, 4, 4, self.base_channels*16))

        # apply CBN
        x = self.cbn(x, z_k, training)

        output = self.relu(x)
        return output


##############################
# Residual Up-sampling Block #
##############################

class ResidualUpsamplingBlock(Layer):
    def __init__(self, init_gain, noise_dim, base_channels, mask_scale, output_channels):
        super(ResidualUpsamplingBlock, self).__init__()

        self.upsample = UpSampling2D(size=(2, 2), interpolation='bilinear')  # up-sampling layer
        # perform 1x1 convolutions on the identity to adjust the number of channels to the output of the conputational
        # pipeline
        self.process_identity = SpectralNormalization(Conv2D(filters=base_channels*output_channels, kernel_size=(1, 1),
                                                    padding='same', kernel_initializer=orthogonal(gain=init_gain)))
        # apply max-pooling to down-sample to segmentation mask
        self.mask_pool = AveragePooling2D(pool_size=mask_scale, padding='same')

        # computational pipeline
        self.conv1 = SpectralNormalization(Conv2D(filters=base_channels*output_channels, kernel_size=(3, 3), padding='same',
                                       kernel_initializer=orthogonal(gain=init_gain)))

        self.cbn1 = ConditionalBatchNormalization(noise_dim, 256, base_channels*output_channels)
        self.relu = ReLU()
        self.conv2 = SpectralNormalization(Conv2D(filters=base_channels*output_channels, kernel_size=(3, 3), padding='same',
                                       kernel_initializer=orthogonal(gain=init_gain)))
        self.cbn2 = ConditionalBatchNormalization(noise_dim, 256, base_channels*output_channels)

    @tf.function
    def call(self, x, z_k, mask, training):

        # resize mask to fit input shape
        mask = tf.cast(self.mask_pool(mask), tf.float32)

        # concatenate mask
        x = tf.concat((x, mask), axis=3)

        # up-sample input
        x = self.upsample(x)

        # save identity
        identity = self.process_identity(x, training)

        # pass input through pipeline
        x = self.conv1(x, training)
        x = self.cbn1(x, z_k, training)
        x = self.relu(x)
        x = self.conv2(x, training)
        x = self.cbn2(x, z_k, training)

        # skip-connection
        x += identity

        # apply ReLU activation
        x = self.relu(x)

        return x


###############
# Final Block #
###############

class OutputBlock(Layer):
    def __init__(self, init_gain, noise_dim, base_channels, output_channels):
        super(OutputBlock, self).__init__()
        self.cbn = ConditionalBatchNormalization(noise_dim, 256, base_channels*output_channels)
        self.relu = ReLU()
        self.conv = SpectralNormalization(Conv2D(filters=3, kernel_size=(3, 3), padding='same', kernel_initializer=orthogonal(gain=init_gain)))

    @tf.function
    def call(self, x, z_k, masks, training):
        x = self.cbn(x, z_k, training)
        x = self.relu(x)
        x = tf.concat((x, tf.cast(masks, tf.float32)), axis=3)
        x = self.conv(x, training)
        x = tf.keras.activations.tanh(x)

        return x


#####################
# Generator Network #
#####################

class Generator(Model):
    def __init__(self, init_gain, input_dim=32, base_channels=32):
        super(Generator, self).__init__()

        # set name for model saving
        self.model_name = 'Generator'

        self.k = None  # region for which the generator is trained
        self.base_channels = base_channels  # data-dependent constant used for number of channels throughout the network

        # first computational block | fully-connected layer + reshaping of noise vector to allow for convolutions
        self.block_1 = InputBlock(init_gain=init_gain,
                                  input_dim=input_dim, base_channels=self.base_channels, output_channels=16)

        # second computational block | residual up-sampling layers
        # mask_scale: down-scaling factor of segmentation mask to be concatenated to the feature maps
        # output_channels: factor by which to multiply base_channels to get final number of feature maps
        self.up_res_block_1 = ResidualUpsamplingBlock(init_gain=init_gain, noise_dim=input_dim, base_channels=self.base_channels,
                                                      mask_scale=32, output_channels=16)
        self.up_res_block_2 = ResidualUpsamplingBlock(init_gain=init_gain, noise_dim=input_dim, base_channels=self.base_channels,
                                                      mask_scale=16, output_channels=8)
        self.up_res_block_3 = ResidualUpsamplingBlock(init_gain=init_gain, noise_dim=input_dim, base_channels=self.base_channels,
                                                      mask_scale=8, output_channels=4)
        self.up_res_block_4 = ResidualUpsamplingBlock(init_gain=init_gain, noise_dim=input_dim, base_channels=self.base_channels,
                                                      mask_scale=4, output_channels=2)

        # computational block 3 | self-attention module
        self.block_3 = SelfAttentionModule(init_gain=init_gain, output_channels=2*base_channels)

        # computational block 4 | final residual up-sampling block
        self.block_4 = ResidualUpsamplingBlock(init_gain=init_gain, noise_dim=input_dim, base_channels=self.base_channels, mask_scale=2,
                                               output_channels=1)

        # computational block 5 | output block
        self.block_5 = OutputBlock(init_gain=init_gain, noise_dim=input_dim, base_channels=self.base_channels, output_channels=1)

    def call(self, batch_images_real, batch_masks_logits, noise_dim, training):
        """
        Forward pass of the generator network. Create a batch of fake images.
        :param batch_images_real: batch of images taken from the dataset
        :param batch_masks_logits: extracted segmentation masks | shape: [batch_size, height, width, n_classes-1]
        :param noise_dim: dimensionality of the noise vector
        :param training: current network phase to switch between modes for CBN layers
        :return: batch of fake images | shape: [batch_size, height, width, 3]
        """

        # number of different regions
        n_regions = batch_masks_logits.shape[3]

        # sample noise vector
        z_k = tf.random.normal([batch_masks_logits.shape[0], noise_dim])

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

        return batch_images_fake, z_k

    def set_region(self, k):
        self.k = k

    def set_name(self, name):
        self.model_name = name


if __name__ == '__main__':

    init_gain = 0.8
    generator = Generator(init_gain=init_gain)
    generator.set_region(0)
    optimizer = Adam(learning_rate=1e-1, beta_1=0, beta_2=0.9)
    input = tf.random.normal([1, 128, 128, 3])
    mask = tf.random.uniform([1, 128, 128, 2], 0, 1, tf.float32)
    with tf.GradientTape() as tape:
        output, z_k = generator(input, mask, noise_dim=32, training=True)
    gradients = tape.gradient(output, generator.trainable_variables)
    # update weights
    optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    output = output[0].numpy()
    output -= np.min(output)
    output /= (np.max(output) - np.min(output))
    plt.imshow(output)
    plt.show()
