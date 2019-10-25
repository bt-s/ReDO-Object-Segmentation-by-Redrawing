#!/usr/bin/python3

"""datasets.py - Handling of supported datasets. For getting the datasets see
                 `get_datasets.py`

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chmielewski-Anders, Mats Steinweg & Bas Straathof"


import tensorflow as tf

from tensorflow.keras.optimizers import Adam

import generator
from discriminator import Discriminator
from train_utils import UnsupervisedLoss

if __name__ == '__main__':
    # Create generator object
    generator = generator.Generator(n_classes=2, n_input=32, base_channels=32,
            init_gain=1.0)

    # Discriminator network
    discriminator = Discriminator(init_gain=1.0)

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
    batch_image_real = tf.expand_dims(tf.image.per_image_standardization(
        image_real), 0)
    mask = tf.image.decode_jpeg(tf.io.read_file(label_path_1), channels=1)
    mask = tf.expand_dims(tf.image.resize(mask, (128, 128),
        preserve_aspect_ratio=False), 0)
    background_color = 29
    mask = tf.expand_dims(tf.cast(tf.where(tf.logical_or(mask <= 0.9 * \
            background_color, mask >= 1.1 * background_color), 1, 0),
            tf.float32)[:, :, :, 0], 3)
    masks = tf.concat((mask, 1-mask), axis=3)

    with tf.GradientTape() as tape:
        batch_image_fake = generator(batch_image_real, masks,
                update_generator=False, training=True)
        d_logits_fake = discriminator(batch_image_fake, training=True)
        d_logits_real = discriminator(batch_image_real, training=True)

        d_loss_r, d_loss_f = loss.get_d_loss(d_logits_real, d_logits_fake)
        d_loss = d_loss_r + d_loss_f
        print('D_R: ', d_loss_r)
        print('D_F: ', d_loss_f)

    gradients = tape.gradient(d_loss, discriminator.trainable_variables)
    # Update weights
    optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

