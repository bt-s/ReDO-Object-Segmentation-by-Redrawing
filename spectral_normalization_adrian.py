import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import layers
from train_utils import UnsupervisedLoss
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import Orthogonal
from network_components import SpectralNormalization
import numpy as np

np.random.seed(10)
W = np.random.randn(3, 3, 1, 1)
W_tf = tf.convert_to_tensor(W)
print(W_tf)


def make_discriminator_model():
  model = tf.keras.Sequential()
  model.add(SpectralNormalization(layers.Conv2D(1, kernel_size=(3,3), strides=(1,1), padding='same',
                          kernel_initializer=Orthogonal(gain=1.0), input_shape=(128, 128, 3))))
  return model


if __name__ == '__main__':

    # fake and real image
    generated_image = tf.ones([1, 3, 3, 1])
    input_image = tf.ones([1, 3, 3, 1])

    # loss and optimizer
    loss = UnsupervisedLoss(lambda_z=5)
    optimizer = SGD(learning_rate=0.1)

    # create network
    discriminator = make_discriminator_model()
    _ = discriminator(input_image)
    discriminator.layers[0].weights[0] = W_tf

    # perform forward pass
    with tf.GradientTape() as tape:
        d_logits_real = discriminator(input_image)
        discriminator.layers[0].reassign_orig()
        d_logits_fake = discriminator(generated_image)
        d_loss_r, d_loss_f = loss.get_d_loss(d_logits_real, d_logits_fake)
        d_loss = d_loss_r + d_loss_f
    gradients = tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator.layers[0].reassign_orig()
    print('Gradients: ', gradients[0][:, 0, 0, :])
    tf.debugging.assert_equal(discriminator.layers[0].weights[0], discriminator.layers[0].W_orig)
    W_old = tf.identity(discriminator.layers[0].W_orig)
    optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    tf.debugging.assert_equal(discriminator.layers[0].W_orig, W_old)
    tf.debugging.assert_equal(discriminator.layers[0].weights[0], W_old-0.1*gradients[0])
    print('W_post_update: ', discriminator.layers[0].weights[0][:, 0, 0, :])

    with tf.GradientTape() as tape:
        d_logits_real = discriminator(input_image)
        discriminator.layers[0].reassign_orig()
        d_logits_fake = discriminator(generated_image)
        d_loss_r, d_loss_f = loss.get_d_loss(d_logits_real, d_logits_fake)
        d_loss = d_loss_r + d_loss_f
    gradients = tape.gradient(d_loss, discriminator.trainable_variables)
    print('Gradients: ', gradients[0][0, :, :, 0])
    discriminator.layers[0].reassign_orig()
    optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    print('W_post_update: ', discriminator.layers[0].weights[0][:, 0, 0, :])
    print('Output:', d_logits_fake)
