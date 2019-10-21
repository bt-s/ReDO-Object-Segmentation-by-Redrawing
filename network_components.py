import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, ReLU, Conv2D, MaxPool2D, Softmax, AveragePooling2D
from tensorflow.keras.initializers import orthogonal
from tensorflow.keras.constraints import Constraint

################################
# Spectral Normalization Layer #
################################

class SpectralNormalization(Layer):
    def __init__(self, layer, n_power_iterations=1):
        """
        Spectral Normalization Layer to wrap around a Conv2D Layer. Kernel Weights are normalized before each forward
        pass.
        :param layer: Conv2D object
        :param n_power_iterations: number of power iterations | default: 1
        """
        super(SpectralNormalization, self).__init__()
        self.layer = layer
        self.init = False  # Conv2D layer's weights haven't been initialized yet
        self.n_power_iterations = n_power_iterations
        self.u = None  # u cannot be initialized here, since kernel shape is not known yet
        self.W_orig = None

    def normalize_weights(self, training=True):
        """
        Normalize the Conv2D layer's weights w.r.t. their spectral norm.
        :param training: True if model in training phase. Updates estimate of u at every iteration.
        """

        filters = self.layer.weights[0].shape.as_list()[-1]  # number of filter kernels in Conv2D layer

        W_orig = self.layer.kernel_orig
        # reshape kernel weights
        W_res = tf.reshape(W_orig, [filters, -1])

        # compute spectral norm and singular value approximation
        spectral_norm, u = self.power_iteration(W_res)

        # normalize kernel weights
        W_sn = W_orig / spectral_norm

        # update estimate of singular vector during training
        if training:
            self.u = u

        return W_sn

    def power_iteration(self, W, n_iter=1):
        """
        Compute approximate spectral norm. According to paper n_iter = 1 is sufficient due to updated u.
        :param W: Reshaped kernel weights | shape: [filters, N]
        :param n_iter: number of power iterations
        :return: approximate spectral norm and updated singular vector approximation.
        """
        if self.u is None:
            self.u = tf.Variable(tf.random.normal([self.layer.weights[0].shape.as_list()[-1], 1]), trainable=False)

        for _ in range(n_iter):
            v = self.normalize_l2(tf.matmul(W, self.u, transpose_a=True))
            u = self.normalize_l2(tf.matmul(W, v))
            print('U:', u)
            print('V: ', v)
            spectral_norm = tf.matmul(tf.matmul(u, W, transpose_a=True), v)
            print('Sigma: ', spectral_norm)

        return spectral_norm, u

    @staticmethod
    def normalize_l2(v, epsilon=1e-12):
        """
        Normalize input matrix w.r.t. its euclidean norm
        :param v: input matrix of arbitrary shape
        :param epsilon: small epsilon to avoid division by zero
        :return: l2-normalized input matrix
        """

        return v / (tf.math.reduce_sum(v ** 2) ** 0.5 + epsilon)

    def call(self, x, training):

        # perform forward pass of Conv2D layer on first iteration to initialize weights
        if not self.init:
            _ = self.layer(x)
            self.layer.kernel_orig = self.add_weight('kernel_orig', self.layer.kernel.shape, trainable=True)
            weights = self.layer.get_weights()
            self.layer.set_weights([tf.identity(weights[0]), weights[0]])
            self.init = True

        # normalize weights before performing standard forward pass of Conv2D layer
        W_sn = self.normalize_weights(training=training)
        self.layer.kernel = W_sn
        output = self.layer(x)
        return output


#########################
# Self-Attention Module #
#########################

class SelfAttentionModule(Layer):
    def __init__(self, init_gain, output_channels, key_size=None):
        super(SelfAttentionModule, self).__init__()

        # set number of key channels
        if key_size is None:
            self.key_size = output_channels // 8
        else:
            self.key_size = key_size

        # trainable parameter to control influence of learned attention maps
        self.gamma = tf.Variable(0.0, name='self_attention_gamma')

        # learned transformation
        self.f = SpectralNormalization(Conv2D(filters=self.key_size, kernel_size=(1, 1), kernel_initializer=orthogonal(gain=init_gain)))
        self.g = SpectralNormalization(Conv2D(filters=self.key_size, kernel_size=(1, 1), kernel_initializer=orthogonal(gain=init_gain)))
        self.h = SpectralNormalization(Conv2D(filters=output_channels, kernel_size=(1, 1), kernel_initializer=orthogonal(gain=init_gain)))
        self.out = SpectralNormalization(Conv2D(filters=output_channels, kernel_size=(1, 1), kernel_initializer=orthogonal(gain=init_gain)))

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

    def call(self, x, training):

        H, W, C = x.shape.as_list()[1:]  # width, height, channel

        # compute query, key and value matrices
        Q = tf.reshape(self.f(x, training), [-1, H * W, self.key_size])
        K = tf.reshape(self.g(x, training), [-1, H * W, self.key_size])
        V = tf.reshape(self.h(x, training), [-1, H * W, C])

        # compute attention maps
        o = self.compute_attention(Q, K, V)
        o = tf.reshape(o, [-1, H, W, C])
        o = self.out(o, training)

        # add weighted attention maps to input feature maps
        output = self.gamma * o + x
        #print('SA_Gamma: ', self.gamma)

        return output
