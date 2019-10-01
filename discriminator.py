import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, ReLU, Conv2D, MaxPool2D, Softmax, AveragePooling2D
from tensorflow.keras.initializers import orthogonal


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
        self.process_identity = Conv2D(filters=output_channels, kernel_size=(1, 1), strides=stride,
                                       kernel_initializer=orthogonal(gain=init_gain))

        # residual pipeline
        self.conv_1 = Conv2D(filters=output_channels, kernel_size=(3, 3), strides=stride, padding='same',
                                       kernel_initializer=orthogonal(gain=init_gain))
        self.in_1 = LayerNormalization(axis=(1, 2))
        self.relu = ReLU()
        self.conv_2 = Conv2D(filters=output_channels, kernel_size=(3, 3), padding='same',
                                       kernel_initializer=orthogonal(gain=init_gain))
        self.in_2 = LayerNormalization(axis=(1, 2))

    @tf.function
    def call(self, x):

        # save identity
        identity = self.process_identity(x)

        # pass input through pipeline
        x = self.conv_1(x)
        x = self.in_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.in_2(x)

        # skip-connection
        x += identity

        # apply ReLU activation
        x = self.relu(x)

        return x


#########################
# Self-Attention Module #
#########################

class SelfAttentionModule(Layer):
    """
    Self-Attention Module. Adapt feature maps to allow for modelling long-range dependencies.
    """
    def __init__(self, init_gain, output_channels, key_size=None):
        """
        :param init_gain: initializer gain for orthogonal initialization
        :param output_channels: number of output channels
        :param key_size: number of channels for the computation of attention maps | default: output_channels // 8
        """
        super(SelfAttentionModule, self).__init__()

        # set number of key channels
        if key_size is None:
            self.key_size = output_channels // 8
        else:
            self.key_size = key_size

        # trainable parameter to control influence of learned attention maps
        self.gamma = tf.Variable(0.0, name='self_attention_gamma')

        # learned transformation
        self.f = Conv2D(filters=self.key_size, kernel_size=(1, 1), kernel_initializer=orthogonal(gain=init_gain))
        self.g = Conv2D(filters=self.key_size, kernel_size=(1, 1), kernel_initializer=orthogonal(gain=init_gain))
        self.h = Conv2D(filters=output_channels, kernel_size=(1, 1), kernel_initializer=orthogonal(gain=init_gain))
        self.out = Conv2D(filters=output_channels, kernel_size=(1, 1), kernel_initializer=orthogonal(gain=init_gain))

    @staticmethod
    def compute_attention(Q, K, V):
        """
        Compute attention maps from queries, keys and values.
        :param Q: Queries
        :param K: Keys
        :param V: Values
        :return: attention map with same shape as input feature maps
        """

        dot_product = tf.matmul(Q, K, transpose_b=True)
        weights = Softmax(axis=2)(dot_product)
        x = tf.matmul(weights, V)
        return x

    def call(self, x):

        H, W, C = x.shape.as_list()[1:]  # width, height, channel

        # compute query, key and value matrices
        Q = tf.reshape(self.f(x), [-1, H*W, self.key_size])
        K = tf.reshape(self.g(x), [-1, H*W, self.key_size])
        V = tf.reshape(self.h(x), [-1, H*W, C])

        # compute attention maps
        o = self.compute_attention(Q, K, V)
        o = tf.reshape(o, [-1, H, W, C])
        o = self.out(o)

        # add weighted attention maps to input feature maps
        output = self.gamma * o + x

        return output


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
        self.block_3 = Sequential((self.res_block_2, self.res_block_3, self.res_block_4, self.res_block_5,
                                   self.res_block_6))

        # spatial sum pooling
        self.block_4 = AveragePooling2D(pool_size=(4, 4), padding='same')

        # dense classification layer
        self.block_5 = Dense(units=1, kernel_initializer=orthogonal(gain=init_gain))

    @tf.function
    def call(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
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

    init_gain = 1.0
    generator = Discriminator(init_gain=init_gain)
    input = tf.random.normal([10, 128, 128, 3])
    output = generator(input)
    print(output)
