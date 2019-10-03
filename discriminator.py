import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, ReLU, Conv2D, MaxPool2D, Softmax, AveragePooling2D
from tensorflow.keras.initializers import orthogonal
from network_components import SelfAttentionModule, SpectralNormalization
from generator import Generator

##################
# Residual Block #
##################

class ResidualBlock(Layer):
    """
    Residual computation block with down-sampling option and variable number of output channels.
    """
    def __init__(self, init_gain, stride, output_channels):
        """
        :param init_gain: initializer gain for orthogonal initialization
        :param stride: stride of the convolutional layers | tuple or int
        :param output_channels: number of output channels
        """
        super(ResidualBlock, self).__init__()

        # perform 1x1 convolutions on the identity to adjust the number of channels to the output of the residual
        # pipeline
        self.process_identity = SpectralNormalization(Conv2D(filters=output_channels, kernel_size=(1, 1), strides=stride,
                                       kernel_initializer=orthogonal(gain=init_gain)))

        # residual pipeline
        self.conv_1 = SpectralNormalization(Conv2D(filters=output_channels, kernel_size=(3, 3), strides=stride, padding='same',
                                       kernel_initializer=orthogonal(gain=init_gain)))
        self.relu = ReLU()
        self.conv_2 = SpectralNormalization(Conv2D(filters=output_channels, kernel_size=(3, 3), padding='same',
                                       kernel_initializer=orthogonal(gain=init_gain)))

    @tf.function
    def call(self, x, training):

        # save identity
        identity = self.process_identity(x, training)

        # pass input through pipeline
        x = self.conv_1(x, training)
        x = self.relu(x)
        x = self.conv_2(x, training)

        # skip-connection
        x += identity

        # apply ReLU activation
        x = self.relu(x)

        return x


#########################
# Discriminator Network #
#########################

class Discriminator(Model):
    def __init__(self, init_gain):
        super(Discriminator, self).__init__()

        # set model's name
        self.model_name = 'Discriminator'

        # input residual down-sampling block
        self.block_1 = ResidualBlock(init_gain=init_gain, output_channels=64,
                                     stride=(2, 2))

        # self-attention module
        self.block_2 = SelfAttentionModule(init_gain=init_gain, output_channels=64)

        # sequence of residual down-sampling blocks
        self.res_block_2 = ResidualBlock(init_gain=init_gain, output_channels=64, stride=(2, 2))
        self.res_block_3 = ResidualBlock(init_gain=init_gain, output_channels=128, stride=(2, 2))
        self.res_block_4 = ResidualBlock(init_gain=init_gain, output_channels=256, stride=(2, 2))
        self.res_block_5 = ResidualBlock(init_gain=init_gain, output_channels=512, stride=(2, 2))
        self.res_block_6 = ResidualBlock(init_gain=init_gain, output_channels=1024, stride=(1, 1))

        # spatial sum pooling
        self.block_4 = AveragePooling2D(pool_size=(4, 4), padding='same')

        # dense classification layer
        self.block_5 = Dense(units=1, kernel_initializer=orthogonal(gain=init_gain))

    @tf.function
    def call(self, x, training):
        x = self.block_1(x, training)
        x = self.block_2(x, training)
        x = self.res_block_2(x, training)
        x = self.res_block_3(x, training)
        x = self.res_block_4(x, training)
        x = self.res_block_5(x, training)
        x = self.res_block_6(x, training)
        x = self.block_4(x)[:, 0, 0, :] * self.block_4.pool_size[0] * self.block_4.pool_size[1]
        x = self.block_5(x)
        return x

    def set_name(self, name):
        """
        Set name of the model.
        :param name: string containing model name
        """

        self.model_name = name


if __name__ == '__main__':

    discriminator = Discriminator(init_gain=0.8)
    generator = Generator(init_gain=1.0)
    generator.set_region(0)
    input_path_1 = 'Datasets/Birds/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
    label_path_1 = 'Datasets/Birds/labels/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.png'
    image_real = tf.image.decode_jpeg(tf.io.read_file(input_path_1))
    image_real = tf.image.resize(image_real, (128, 128), preserve_aspect_ratio=False)
    image_real = tf.expand_dims(tf.image.per_image_standardization(image_real), 0)
    mask = tf.image.decode_png(tf.io.read_file(label_path_1))
    mask = tf.expand_dims(tf.image.resize(mask, (128, 128), preserve_aspect_ratio=False), 0)
    mask = (mask / 255.0 * 2) - 1.0
    mask = tf.concat((mask, -1 * mask), axis=3)
    image_fake, z_k = generator(image_real, mask, noise_dim=32, training=True)

    d_logits_real = discriminator(image_real, training=True)
    d_logits_fake = discriminator(image_fake, training=True)
    print('Logits real: ', d_logits_real)
    print('Logits fake: ', d_logits_fake)
