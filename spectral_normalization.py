import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import layers
from train_utils import UnsupervisedLoss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Orthogonal

class Spectral_Norm(Constraint):
    '''
    Uses power iteration method to calculate a fast approximation
    of the spectral norm (Golub & Van der Vorst)
    The weights are then scaled by the inverse of the spectral norm
    '''
    def __init__(self, power_iters=50):
        self.n_iters = power_iters

    def __call__(self, w):
      flattened_w = tf.reshape(w, [w.shape[0], -1])
      u = tf.random.normal([flattened_w.shape[0]])
      v = tf.random.normal([flattened_w.shape[1]])
      for i in range(self.n_iters):
        v = tf.linalg.matvec(tf.transpose(flattened_w), u)
        v = self.l2_normalize(v)
        u = tf.linalg.matvec(flattened_w, v)
        u = self.l2_normalize(u)
      sigma = tf.tensordot(u, tf.linalg.matvec(flattened_w, v), axes=1)
      print(sigma)
      return w / sigma

    @staticmethod
    def l2_normalize(x, eps=1e-12):
        '''
        Scale input by the inverse of it's euclidean norm
        '''
        return x / tf.linalg.norm(x + eps)

    def get_config(self):
        return {'n_iters': self.n_iters}


def make_discriminator_model():
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same',
                          kernel_initializer=Orthogonal(gain=1.0),
                          kernel_constraint=Spectral_Norm(), input_shape=(128, 128, 3)))
  model.add(layers.LeakyReLU(0.1))
  model.add(layers.Conv2D(64, kernel_size=(4,4), strides=(2,2), padding='same',
                          kernel_initializer=Orthogonal(gain=1.0),
                          kernel_constraint=Spectral_Norm()))
  model.add(layers.LeakyReLU(0.1))
  model.add(layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same',
                          kernel_initializer=Orthogonal(gain=1.0),
                          kernel_constraint=Spectral_Norm()))
  model.add(layers.LeakyReLU(0.1))
  model.add(layers.Conv2D(128, kernel_size=(4,4), strides=(2,2), padding='same',
                          kernel_initializer=Orthogonal(gain=1.0),
                          kernel_constraint=Spectral_Norm()))
  model.add(layers.LeakyReLU(0.1))
  model.add(layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same',
                          kernel_initializer=Orthogonal(gain=1.0),
                          kernel_constraint=Spectral_Norm()))
  model.add(layers.LeakyReLU(0.1))
  model.add(layers.Conv2D(256, kernel_size=(4,4), strides=(2,2), padding='same',
                          kernel_initializer=Orthogonal(gain=1.0),
                          kernel_constraint=Spectral_Norm()))
  model.add(layers.LeakyReLU(0.1))
  model.add(layers.Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='same',
                          kernel_initializer=Orthogonal(gain=1.0),
                          kernel_constraint=Spectral_Norm()))
  model.add(layers.LeakyReLU(0.1))
  model.add(layers.Flatten())
  model.add(layers.Dense(1, kernel_constraint=Spectral_Norm()))
  return model


if __name__ == '__main__':

    generated_image = tf.random.normal([2, 128, 128, 3])
    input_image = tf.random.normal([2, 128, 128, 3])
    loss = UnsupervisedLoss(lambda_z=5)
    optimizer = Adam(learning_rate=0, beta_1=0, beta_2=0.9)
    sn = Spectral_Norm()
    discriminator = make_discriminator_model()
    print(discriminator.layers[0].weights[0][:, :, 0, 0])
    with tf.GradientTape() as tape:
        d_logits_real = discriminator(input_image)
        d_logits_fake = discriminator(generated_image)
        d_loss_r, d_loss_f = loss.get_d_loss(d_logits_real, d_logits_fake)
        d_loss = d_loss_r + d_loss_f
    print(discriminator.layers[0].weights[0][:, :, 0, 0])
    gradients = tape.gradient(d_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    print(discriminator.layers[0].weights[0][:, :, 0, 0])

    with tf.GradientTape() as tape:
        d_logits_real = discriminator(input_image)
        d_logits_fake = discriminator(generated_image)
        d_loss_r, d_loss_f = loss.get_d_loss(d_logits_real, d_logits_fake)
        d_loss = d_loss_r + d_loss_f
    print(discriminator.layers[0].weights[0][:, :, 0, 0])
    gradients = tape.gradient(d_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    print(discriminator.layers[0].weights[0][:, :, 0, 0])

    print("decision: ", d_logits_fake)