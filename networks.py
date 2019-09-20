import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D, LayerNormalization, ReLU, AveragePooling2D, \
    UpSampling2D, Softmax


#######################
# Convolutional Block #
#######################

class ConvolutionalBlock(Model):
    def __init__(self, filters, kernel_size, padding, stride):
        super(ConvolutionalBlock, self).__init__()
        self.block = tf.keras.Sequential()
        self.block.add(Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=stride))
        self.block.add(LayerNormalization(axis=(1, 2, 3)))
        self.block.add(ReLU())

    @tf.function
    def call(self, x):

        return self.block(x)


###########################
# Pyramid Pooling Module #
###########################

class PPM(Model):
    def __init__(self, input_dim=(32, 32)):
        super(PPM, self).__init__()

        # scale 1 (1x1 Output)
        pool_size_1 = (input_dim[0] // 1, input_dim[1] // 1)
        self.avg_pool_1 = AveragePooling2D(pool_size_1)
        self.conv_1 = Conv2D(filters=1, kernel_size=(1, 1), padding='same')
        self.upsample_1 = UpSampling2D(size=pool_size_1, interpolation='bilinear')

        # scale 2 (2x2 Output)
        pool_size_2 = (input_dim[0] // 2, input_dim[1] // 2)
        self.avg_pool_2 = AveragePooling2D(pool_size_2)
        self.conv_2 = Conv2D(filters=1, kernel_size=(1, 1), padding='same')
        self.upsample_2 = UpSampling2D(size=pool_size_2, interpolation='bilinear')

        # scale 3 (4x4 Output)
        pool_size_3 = (input_dim[0] // 4, input_dim[1] // 4)
        self.avg_pool_3 = AveragePooling2D(pool_size_3)
        self.conv_3 = Conv2D(filters=1, kernel_size=(1, 1), padding='same')
        self.upsample_3 = UpSampling2D(size=pool_size_3, interpolation='bilinear')

        # scale 4 (8x8 Output)
        pool_size_4 = (input_dim[0] // 8, input_dim[1] // 8)
        self.avg_pool_4 = AveragePooling2D(pool_size_4)
        self.conv_4 = Conv2D(filters=1, kernel_size=(1, 1), padding='same')
        self.upsample_4 = UpSampling2D(size=pool_size_4, interpolation='bilinear')

        # final up-sampling
        self.upsample = UpSampling2D(size=(2, 2), interpolation='bilinear')

    @tf.function
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
        x = self.upsample(x)

        return x


##################
# Residual Block #
##################

class ResidualBlock(Model):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.in1 = LayerNormalization(axis=(1, 2, 3))
        self.relu = ReLU()
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.in2 = LayerNormalization(axis=(1, 2, 3))

    @tf.function
    def call(self, x):

        # store input for skip-connection
        identity = x

        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.in2(x)

        # skip-connection
        x += identity

        # apply ReLU activation
        x = self.relu(x)

        return x


######################
# Reflection Padding #
######################

class ReflectionPadding2D(Layer):
    """
    Reflection padding layer
    """
    def __init__(self, padding=(3, 3)):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__()

    def output_shape(self, x):
        # for option "channel_last"
        return x[0], x[1] + 2 * self.padding[0], x[2] + 2 * self.padding[1], x[3]

    @tf.function
    def call(self, x):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


##################
# Mask Generator #
##################

class MaskGenerator(Model):
    def __init__(self, n_classes):
        super(MaskGenerator, self).__init__()

        self.n_classes = n_classes  # number of output classes including background

        # first computational block (3 convolutional layers)
        self.ref_padding_1 = ReflectionPadding2D(padding=(3, 3))
        self.conv_block_1 = ConvolutionalBlock(filters=16, kernel_size=(7, 7), padding='valid', stride=1)
        self.conv_block_2 = ConvolutionalBlock(filters=32, kernel_size=(3, 3), padding='same', stride=2)
        self.conv_block_3 = ConvolutionalBlock(filters=64, kernel_size=(3, 3), padding='same', stride=2)
        self.block_1 = Sequential((self.ref_padding_1, self.conv_block_1, self.conv_block_2, self.conv_block_3))

        # second computational block (3 residual blocks)
        self.res_block_1 = ResidualBlock()
        self.res_block_2 = ResidualBlock()
        self.res_block_3 = ResidualBlock()
        self.block_2 = tf.keras.Sequential((self.res_block_1, self.res_block_2, self.res_block_3))

        # third computational block (1 Pyramid Pooling Module)
        self.block_3 = Sequential(PPM(input_dim=(32, 32)))

        # fourth computational block (1 convolutional layer, 1 up-sampling layer, 2 convolutional layers)
        self.conv_block_4 = ConvolutionalBlock(filters=34, kernel_size=(3, 3), padding='same', stride=1)
        self.upsample = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_block_5 = ConvolutionalBlock(filters=17, kernel_size=(3, 3), padding='same', stride=1)
        self.ref_padding_2 = ReflectionPadding2D(padding=(3, 3))
        self.conv_block_6 = ConvolutionalBlock(filters=self.n_classes, kernel_size=(7, 7), padding='valid', stride=1)
        self.block_4 = Sequential((self.conv_block_4, self.upsample, self.conv_block_5, self.ref_padding_2,
                                   self.conv_block_6))

    @tf.function
    def __call__(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        # compute output depending on number of classes
        if self.n_classes == 2:
            output = tf.math.sigmoid(x)
        else:
            output = Softmax(axis=3)(x)

        return output


if __name__ == '__main__':

    input_path_1 = 'Datasets/Flowers/images/image_00083.jpg'
    image_1 = tf.image.decode_jpeg(tf.io.read_file(input_path_1))
    image_1 = tf.image.resize(image_1, (128, 128), preserve_aspect_ratio=False, antialias=True)
    image_1 = tf.expand_dims(image_1, 0)
    input_path_2 = 'Datasets/Flowers/images/image_00081.jpg'
    image_2 = tf.image.decode_jpeg(tf.io.read_file(input_path_2))
    image_2 = tf.image.resize(image_2, (128, 128), preserve_aspect_ratio=False, antialias=True)
    image_2 = tf.expand_dims(image_2, 0)

    image_batch = tf.concat((image_1, image_2), 0)
    f = MaskGenerator(n_classes=2)
    output = f(x=image_batch)
    print(f.trainable_variables)
    print('Output Shape: ', output.shape)
