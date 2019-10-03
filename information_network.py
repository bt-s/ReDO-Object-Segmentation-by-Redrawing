import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Conv2D, MaxPool2D, Softmax, AveragePooling2D
from discriminator import ResidualBlock, SelfAttentionModule
from tensorflow.keras.initializers import orthogonal
from generator import Generator

####################################
# Information Conservation Network #
####################################

class InformationConservationNetwork(Model):
    def __init__(self, init_gain, n_classes, output_dim):
        """
        :param init_gain: initializer for the kernel weights
        :param n_classes: number of classes in the training scheme. Corresponds to number of network heads
        :param output_dim: dimensionality of the output data | int
        """

        super(InformationConservationNetwork, self).__init__()

        self.model_name = 'Information Network'

        # input residual down-sampling block
        self.block_1 = ResidualBlock(init_gain=init_gain, output_channels=64, stride=(2, 2))

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
        self.final_layers = {str(k): Dense(units=output_dim, kernel_initializer=orthogonal(gain=init_gain)) for k in range(n_classes)}

    def call(self, x, k, training):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.res_block_2(x, training)
        x = self.res_block_3(x, training)
        x = self.res_block_4(x, training)
        x = self.res_block_5(x, training)
        x = self.res_block_6(x, training)
        x = self.block_4(x)[:, 0, 0, :] * self.block_4.pool_size[0] * self.block_4.pool_size[1]
        x = self.final_layers[str(k)](x)
        return x

    def set_name(self, name):
        self.model_name = name


if __name__ == '__main__':

    I = InformationConservationNetwork(init_gain=1.0, n_classes=2, output_dim=32)
    generator = Generator(init_gain=1.0)
    k = 0
    generator.set_region(k=k)
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

    z_k_hat = I(image_fake, k, training=True)
    print('z_k: ', z_k)
    print('z_k_hat: ', z_k_hat)
