import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Conv2D, MaxPool2D, Softmax, AveragePooling2D
from discriminator import ResidualBlock, SelfAttentionModule
from tensorflow.keras.initializers import orthogonal


####################################
# Information Conservation Network #
####################################

class InformationConservationNetwork(Model):
    def __init__(self, init_gain, n_classes, n_output):
        """
        :param init_gain: initializer for the kernel weights
        :param n_classes: number of classes in the training scheme. Corresponds to number of network heads
        :param n_output: dimensionality of the output data | int
        """

        super(InformationConservationNetwork, self).__init__()

        self.model_name = 'Information_Network'
        self.n_classes = n_classes

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
        self.final_layer = Dense(units=self.n_classes*n_output, kernel_initializer=orthogonal(gain=init_gain))

    def call(self, x, training):
        x = self.block_1(x, training)
        x = self.block_2(x, training)
        x = self.res_block_2(x, training)
        x = self.res_block_3(x, training)
        x = self.res_block_4(x, training)
        x = self.res_block_5(x, training)
        x = self.res_block_6(x, training)
        x = self.block_4(x)[:, 0, 0, :] * self.block_4.pool_size[0] * self.block_4.pool_size[1]
        x = self.final_layer(x)
        x = tf.reshape(x, [x.shape[0]*self.n_classes, -1])
        return x


