import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, ReLU, Conv2D, UpSampling2D, MaxPool2D, Softmax
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam


###################################
# Conditional Batch Normalization #
###################################

class ConditionalBatchNormalization(Model):
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

        self.beta = tf.Variable(0.0)  # trainable scale variable
        self.gamma = tf.Variable(1.0)  # trainable shift variable

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
    def __call__(self, x, noise, training):

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
        output = tf.add(tf.multiply(gamma_c, x), beta_c)

        return output


if __name__ == '__main__':

    noise = tf.random.normal([1, 32])
    input = tf.random.uniform([1, 64, 64, 16], 0, 2, tf.float32)
    cbn = ConditionalBatchNormalization(n_input=32, n_hidden=256, n_output=16)

    with tf.GradientTape() as tape:
        output = cbn(input, noise, training=True)
    gradients = tape.gradient(output, cbn.trainable_variables)
    print([x.name for x in cbn.trainable_variables])
    # update weights
    optimizer = Adam(learning_rate=1e-2, beta_1=0, beta_2=0.9)
    optimizer.apply_gradients(zip(gradients, cbn.trainable_variables))
    print(output.shape)
    print(cbn.gamma)
    print(cbn.beta)
    print(gradients)
