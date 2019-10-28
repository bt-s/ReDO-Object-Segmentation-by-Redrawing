#!/usr/bin/python3

"""datasets.py - Handling of supported datasets. For getting the datasets see
                 `get_datasets.py`

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chmielewski-Anders, Mats Steinweg & Bas Straathof"


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.optimizers import Adam

from generator import Generator
from discriminator import Discriminator
from information_network import InformationConservationNetwork
from train_utils import UnsupervisedLoss


if __name__ == '__main__':
    # Create generator object
    generator = Generator(n_classes=2, n_input=32, base_channels=32,
            init_gain=0.8)

    # Discriminator network
    discriminator = Discriminator(init_gain=0.8)

    # Information network
    information_network = InformationConservationNetwork(init_gain=0.8,
            n_classes=2, n_output=32)

    # Create optimizer object
    optimizer = Adam(learning_rate=1e-1, beta_1=0, beta_2=0.9)

    # Create loss function
    loss = UnsupervisedLoss(lambda_z=5.0)

    # Load input image and mask
    input_path_1 = 'Datasets/Flowers/images/image_00001.jpg'
    label_path_1 = 'Datasets/Flowers/labels/label_00001.jpg'
    image_real = tf.image.decode_jpeg(tf.io.read_file(input_path_1))
    image_real = tf.image.resize(image_real, (128, 128),
            preserve_aspect_ratio=False)
    image_real = tf.expand_dims(image_real / 255.0 * 2 - 1, 0)
    mask = tf.image.decode_jpeg(tf.io.read_file(label_path_1), channels=1)
    mask = tf.expand_dims(tf.image.resize(mask, (128, 128),
        preserve_aspect_ratio=False), 0)
    background_color = 29
    mask = tf.expand_dims(tf.cast(tf.where(tf.logical_or(mask <= 0.9 * \
            background_color, mask >= 1.1 * background_color), 10, -10),
            tf.float32)[:, :, :, 0], 3)
    masks = tf.concat((mask, -1*mask), axis=3)
    masks = tf.math.sigmoid(masks)

    with tf.GradientTape() as tape:
        z = tf.random.normal([1, 2, 1, 1, 32])
        batch_image_fake, batch_regions_fake = generator.call(image_real, masks, z, training=True)
        d_logits_fake = discriminator.call(batch_image_fake, training=True)
        z_hat = information_network.call(batch_regions_fake, training=True)
        g_loss_d, g_loss_i = loss.get_g_loss(d_logits_fake, z, z_hat)
        g_loss = g_loss_d + g_loss_i
        print('Generator loss (discriminator): ', g_loss_d)
        print('Generator loss (information): ', g_loss_i)

    gradients = tape.gradient(g_loss,
        generator.trainable_variables)

    # Update weights
    optimizer.apply_gradients(zip(gradients,
        generator.trainable_variables))

    # Input image
    image_real = image_real[0].numpy()
    image_real -= np.min(image_real)
    image_real /= (np.max(image_real) - np.min(image_real))

    # Fake image with redrawn foreground
    image_fake_fg = batch_image_fake[0].numpy()
    image_fake_fg -= np.min(image_fake_fg)
    image_fake_fg /= (np.max(image_fake_fg) - np.min(image_fake_fg))

    # Plot output
    fig, ax = plt.subplots(1, 3)
    ax[0].set_title('Input Image')
    ax[0].imshow(image_real)
    ax[1].set_title('Mask Foreground')
    ax[1].imshow(masks[0].numpy()[:, :, 1], cmap='gray')
    ax[2].set_title('Fake Background')
    ax[2].imshow(image_fake_fg)
    plt.show()


