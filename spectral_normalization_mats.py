import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import layers
from train_utils import UnsupervisedLoss
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import Orthogonal
from network_components import SpectralNormalization

tf.random.set_seed(10)

################################
# Spectral Normalization Layer #
################################


class SpectralNormalization(layers.Layer):
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

        print('W_orig: ', self.layer.weights[0][0, :, :, 0])

        # reshape kernel weights
        W = tf.reshape(self.layer.weights[0], [filters, -1])

        # compute spectral norm and singular value approximation
        spectral_norm, u = self.power_iteration(W)
        print('Sigma: ', spectral_norm)

        # save copy of original weights
        W_orig = tf.identity(self.layer.weights[0])

        # normalize kernel weights
        self.layer.weights[0].assign(self.layer.weights[0] / spectral_norm)
        print('W_sn: ', self.layer.weights[0][0, :, :, 0])

        # update estimate of singular vector during training
        if training:
            self.u = u

        return W_orig

    def power_iteration(self, W, n_iter=40):
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
            spectral_norm = tf.matmul(tf.matmul(u, W, transpose_a=True), v)

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
            self.init = True

        # normalize weights before performing standard forward pass of Conv2D layer
        self.W_orig = self.normalize_weights(training=training)
        output = self.layer(x)

        # re-assign original weights
        self.layer.weights[0].assign(self.W_orig)
        # print('R_orig_reassigned: ', self.layer.weights[0][0, :, :, 0])
        return output

    def reassign_orig(self):
        self.layer.weights[0].assign(self.W_orig)


def make_discriminator_model():
  model = tf.keras.Sequential()
  model.add(SpectralNormalization(layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same',
                          kernel_initializer=Orthogonal(gain=1.0), input_shape=(128, 128, 3))))
  return model


if __name__ == '__main__':

    # fake and real image
    generated_image = tf.random.normal([1, 128, 128, 3])
    input_image = tf.random.normal([1, 128, 128, 3])

    # loss and optimizer
    loss = UnsupervisedLoss(lambda_z=5)
    optimizer = SGD(learning_rate=0.1)

    # create network
    discriminator = make_discriminator_model()

    # perform forward pass
    with tf.GradientTape() as tape:
        d_logits_real = discriminator(input_image)
        d_logits_fake = discriminator(generated_image)
        d_loss_r, d_loss_f = loss.get_d_loss(d_logits_real, d_logits_fake)
        d_loss = d_loss_r + d_loss_f
    gradients = tape.gradient(d_loss, discriminator.trainable_variables)
    print('Gradients: ', gradients[0][:, 0, 0, :])
    tf.debugging.assert_equal(discriminator.layers[0].weights[0], discriminator.layers[0].W_orig)
    W_old = tf.identity(discriminator.layers[0].W_orig)
    optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    tf.debugging.assert_equal(discriminator.layers[0].W_orig, W_old)
    tf.debugging.assert_equal(discriminator.layers[0].weights[0], W_old-0.1*gradients[0])
    print('W_post_update: ', discriminator.layers[0].weights[0][:, 0, 0, :])

    with tf.GradientTape() as tape:
        d_logits_real = discriminator(input_image)
        d_logits_fake = discriminator(generated_image)
        d_loss_r, d_loss_f = loss.get_d_loss(d_logits_real, d_logits_fake)
        d_loss = d_loss_r + d_loss_f
    gradients = tape.gradient(d_loss, discriminator.trainable_variables)
    print('Gradients: ', gradients[0][0, :, :, 0])
    optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    #print('W_post_update: ', discriminator.layers[0].weights[0][:, 0, 0, :])

    [-2.2978231e-04 - 3.8245428e-04  1.2784777e-04]
    [-5.1827502e-04 - 3.5870259e-04  3.9433449e-05]
    [-3.0768802e-04 - 4.4223058e-04 - 3.3559548e-04]], shape = (3, 3), dtype = float32)