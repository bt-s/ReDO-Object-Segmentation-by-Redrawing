import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Conv2D, MaxPool2D, Softmax, AveragePooling2D
from discriminator import ResidualBlock, SelfAttentionModule
from tensorflow.keras.initializers import orthogonal

####################################
# Information Conservation Network #
####################################

class InformationConservationNetwork(Model):
    def __init__(self, init_gain, n_classes, output_dim):
        """
        :param initializer: initializer for the kernel weights
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
        self.block_3 = Sequential((self.res_block_2, self.res_block_3, self.res_block_4, self.res_block_5,
                                   self.res_block_6))

        # spatial sum pooling
        self.block_4 = AveragePooling2D(pool_size=(4, 4), padding='same')

        # dense classification layer
        self.final_layers = {str(k): Dense(units=output_dim, kernel_initializer=orthogonal(gain=init_gain)) for k in range(n_classes)}

    def __call__(self, x, k):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)[:, 0, 0, :] * self.block_4.pool_size[0] * self.block_4.pool_size[1]
        x = self.final_layers[str(k)](x)
        return x

    def set_name(self, name):
        self.model_name = name


if __name__ == '__main__':

    init_gain = 1.0
    generator = InformationConservationNetwork(init_gain=init_gain, n_classes=2, output_dim=32)
    input = tf.random.uniform([1, 128, 128, 3])
    output = generator(input, 0)
    print(output)
