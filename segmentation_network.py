import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization, ReLU, UpSampling2D, Softmax, AveragePooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import orthogonal
from tensorflow.keras.regularizers import L1L2


#######################
# Convolutional Block #
#######################

class ConvolutionalBlock(Model):
    """
    Computational block consisting of a 2D Convolutional layer followed by an Instance Normalization layer and ReLU
    activation.
    """
    def __init__(self, filters, kernel_size, padding, stride, init_gain, weight_decay):
        super(ConvolutionalBlock, self).__init__()
        self.conv_block = Sequential()
        self.conv_block.add(Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=stride,
                            kernel_initializer=orthogonal(gain=init_gain), kernel_regularizer=L1L2(l2=weight_decay)))
        self.conv_block.add(LayerNormalization(axis=(1, 2), center=False, scale=False))
        self.conv_block.add(ReLU())

    def call(self, x):
        x = self.conv_block(x)
        return x


###########################
# Pyramid Pooling Module #
###########################

class PPM(Model):
    """
    Pyramid Pooling Module. Extract features at four different scales and fuse them together.
    """
    def __init__(self, input_shape, init_gain, weight_decay):
        """
        :param init_gain: initialization gain for orthogonal initialization
        :param input_shape: (W, H) of the input feature maps
        :param weight_decay: multiplicative factor for l2 weight regularization
        """
        super(PPM, self).__init__()

        assert len(input_shape) == 3  # check that input_dim tuple provides width and height and number of channels
        # of feature maps

        n_input_channels = input_shape[2]  # number of input channels

        # number of scales at which features are extracted and then concatenated with the original feature maps
        n_scales = 4

        # scale 1 (1x1 Output)
        pool_size_1 = (input_shape[0] // 1, input_shape[1] // 1)
        self.avg_pool_1 = AveragePooling2D(pool_size_1)
        self.conv_1 = Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                             kernel_initializer=orthogonal(gain=init_gain), kernel_regularizer=L1L2(l2=weight_decay))
        self.upsample_1 = UpSampling2D(size=pool_size_1, interpolation='bilinear')

        # scale 2 (2x2 Output)
        pool_size_2 = (input_shape[0] // 2, input_shape[1] // 2)
        self.avg_pool_2 = AveragePooling2D(pool_size_2)
        self.conv_2 = Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                             kernel_initializer=orthogonal(gain=init_gain), kernel_regularizer=L1L2(l2=weight_decay))
        self.upsample_2 = UpSampling2D(size=pool_size_2, interpolation='bilinear')

        # scale 3 (4x4 Output)
        pool_size_3 = (input_shape[0] // 4, input_shape[1] // 4)
        self.avg_pool_3 = AveragePooling2D(pool_size_3)
        self.conv_3 = Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                             kernel_initializer=orthogonal(gain=init_gain), kernel_regularizer=L1L2(l2=weight_decay))
        self.upsample_3 = UpSampling2D(size=pool_size_3, interpolation='bilinear')

        # scale 4 (8x8 Output)
        pool_size_4 = (input_shape[0] // 8, input_shape[1] // 8)
        self.avg_pool_4 = AveragePooling2D(pool_size_4)
        self.conv_4 = Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                             kernel_initializer=orthogonal(gain=init_gain), kernel_regularizer=L1L2(l2=weight_decay))
        self.upsample_4 = UpSampling2D(size=pool_size_4, interpolation='bilinear')

        # final up-sampling
        self.upsample_final = UpSampling2D(size=(2, 2), interpolation='bilinear')

    def call(self, x):

        # scale 1
        x_1 = self.avg_pool_1(x)
        x_1 = self.conv_1(x_1)
        x_1 = self.upsample_1(x_1)

        # scale 2
        x_2 = self.avg_pool_2(x)
        x_2 = self.conv_2(x_2)
        x_2 = self.upsample_2(x_2)

        # scale 3
        x_3 = self.avg_pool_3(x)
        x_3 = self.conv_3(x_3)
        x_3 = self.upsample_3(x_3)

        # scale 4
        x_4 = self.avg_pool_4(x)
        x_4 = self.conv_4(x_4)
        x_4 = self.upsample_4(x_4)

        # concatenate feature maps
        x = tf.concat((x, x_1, x_2, x_3, x_4), 3)

        # up-sample fused features maps
        x = self.upsample_final(x)

        return x


##################
# Residual Block #
##################

class ResidualBlock(Model):
    """
    Residual Computational block. Standard ResNet block using Instance Normalization. ReLU applied to fused feature
    maps.
    """
    def __init__(self, n_channels, init_gain, weight_decay):
        """
        :param n_channels: number of input channels | equal to number of output channels
        :param init_gain: gain for orthogonal initialization
        :param weight_decay: multiplicative factor for l2 weight regularization
        """
        super(ResidualBlock, self).__init__()
        self.conv_1 = Conv2D(filters=n_channels, kernel_size=(3, 3), padding='same', use_bias=False,
                             kernel_initializer=orthogonal(gain=init_gain), kernel_regularizer=L1L2(l2=weight_decay))
        self.in_1 = LayerNormalization(axis=(1, 2))
        self.relu = ReLU()
        self.conv_2 = Conv2D(filters=n_channels, kernel_size=(3, 3), padding='same', use_bias=False,
                             kernel_initializer=orthogonal(gain=init_gain), kernel_regularizer=L1L2(l2=weight_decay))
        self.out = Sequential()
        self.out.add(LayerNormalization(axis=(1, 2)))
        self.out.add(ReLU())

    def call(self, x):

        # store input for skip-connection
        identity = x

        # residual pipeline
        x = self.conv_1(x)
        x = self.in_1(x)
        x = self.relu(x)
        x = self.conv_2(x)

        # skip-connection
        x += identity

        # apply ReLU activation
        x = self.out(x)

        return x


######################
# Reflection Padding #
######################

class ReflectionPadding2D(Layer):
    """
    Reflection padding layer.
    """
    def __init__(self, padding=(3, 3)):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__()

    def call(self, x):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


########################
# Segmentation Network #
########################

class SegmentationNetwork(Model):
    def __init__(self, n_classes, init_gain, weight_decay):
        super(SegmentationNetwork, self).__init__()

        self.model_name = 'Segmentation_Network'  # network name for model saving

        self.n_classes = n_classes  # number of output classes including background

        # first computational block (3 convolutional layers)
        self.ref_padding_1 = ReflectionPadding2D(padding=(3, 3))
        self.conv_block_1 = ConvolutionalBlock(filters=16, kernel_size=(7, 7), padding='valid', stride=1,
                                               init_gain=init_gain, weight_decay=weight_decay)
        self.conv_block_2 = ConvolutionalBlock(filters=32, kernel_size=(3, 3), padding='same', stride=2,
                                               init_gain=init_gain, weight_decay=weight_decay)
        self.conv_block_3 = ConvolutionalBlock(filters=64, kernel_size=(3, 3), padding='same', stride=2,
                                               init_gain=init_gain, weight_decay=weight_decay)
        self.block_1 = Sequential((self.ref_padding_1, self.conv_block_1, self.conv_block_2, self.conv_block_3))

        # second computational block (3 residual blocks)
        self.res_block_1 = ResidualBlock(init_gain=init_gain, n_channels=64, weight_decay=weight_decay)
        self.res_block_2 = ResidualBlock(init_gain=init_gain, n_channels=64, weight_decay=weight_decay)
        self.res_block_3 = ResidualBlock(init_gain=init_gain, n_channels=64, weight_decay=weight_decay)
        self.block_2 = Sequential((self.res_block_1, self.res_block_2, self.res_block_3))

        # third computational block (1 Pyramid Pooling Module)
        self.block_3 = PPM(init_gain=init_gain, input_shape=(32, 32, 64), weight_decay=weight_decay)

        # fourth computational block (1 convolutional layer, 1 up-sampling layer, 2 convolutional layers)
        self.conv_block_4 = ConvolutionalBlock(filters=34, kernel_size=(3, 3), padding='same', stride=1,
                                               init_gain=init_gain, weight_decay=weight_decay)
        self.upsample = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_block_5 = ConvolutionalBlock(filters=17, kernel_size=(3, 3), padding='same', stride=1,
                                               init_gain=init_gain, weight_decay=weight_decay)
        self.ref_padding_2 = ReflectionPadding2D(padding=(3, 3))
        self.conv_final = Conv2D(filters=self.n_classes, kernel_size=(7, 7), padding='valid',
                                kernel_initializer=orthogonal(gain=init_gain), kernel_regularizer=L1L2(l2=weight_decay))
        self.block_4 = Sequential((self.conv_block_4, self.upsample, self.conv_block_5, self.ref_padding_2,
                                   self.conv_final))

    def call(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        output = self.block_4(x)
        return output

    def set_name(self, name):
        """
        Set name of the model for model saving.
        """
        self.model_name = name


if __name__ == '__main__':

    # prepare exemplary image batch
    input_path_1 = 'Datasets/Flowers/images/image_00023.jpg'
    input_path_2 = 'Datasets/Flowers/images/image_00081.jpg'
    image_1 = tf.image.decode_jpeg(tf.io.read_file(input_path_1))
    image_1 = tf.image.resize(image_1, (128, 128), preserve_aspect_ratio=False)
    image_1 = tf.expand_dims(tf.image.per_image_standardization(image_1), 0)
    image_2 = tf.image.decode_jpeg(tf.io.read_file(input_path_2))
    image_2 = tf.image.resize(tf.image.per_image_standardization(image_2), (128, 128), preserve_aspect_ratio=False)
    image_2 = tf.expand_dims(image_2, 0)
    image_batch = tf.concat((image_1, image_2), 0)

    # create network object
    f = SegmentationNetwork(n_classes=2, init_gain=0.8, weight_decay=1e-4)

    # forward pass
    mask_batch = f(image_batch)
    print('Output Shape: ', mask_batch.shape)

    fig, ax = plt.subplots(2, 2)
    image_1 = image_1[0].numpy()
    image_1 -= np.min(image_1)
    image_1 /= (np.max(image_1) - np.min(image_1))
    image_2 = image_2[0].numpy()
    image_2 -= np.min(image_2)
    image_2 /= (np.max(image_2) - np.min(image_2))
    mask_1 = Softmax(axis=2)(mask_batch[0]).numpy()[:, :, 1]
    mask_2 = Softmax(axis=2)(mask_batch[1]).numpy()[:, :, 1]

    # plot images and masks
    ax[0, 0].imshow(image_1)
    ax[1, 0].imshow(image_2)
    ax[0, 1].imshow(mask_1, cmap='gray')
    ax[1, 1].imshow(mask_2, cmap='gray')
    plt.show()

