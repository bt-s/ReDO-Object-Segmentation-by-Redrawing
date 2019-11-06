#!/usr/bin/python3

"""evaluate_masks.py - Script to test models

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chmielewski-Anders, Mats Steinweg & Bas Straathof"


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from redo import SegmentationNetwork
from redo import Generator
from redo.datasets import BirdDataset, FlowerDataset, FaceDataset
from redo import Discriminator

if __name__ == '__main__':
    # Session name
    session_name = 'Unsupervised_Flowers'

    # Iteration to evaluate
    iteration = 3000

    # Create datasets
    dataset = FlowerDataset()
    test_dataset = dataset.get_split('training', batch_size=25)

    # Initializer
    init_gain = 1.0

    # Create loss
    loss = UnsupervisedLoss(lambda_z=5.0)

    # Create model and load weights
    segmentation_network = SegmentationNetwork(n_classes=dataset.n_classes,
            init_gain=init_gain, weight_decay=1e-4)
    generator = Generator(init_gain=init_gain, base_channels=32, n_input=32,
            n_classes=2)
    discriminator = Discriminator(init_gain=init_gain)
    segmentation_network.load_weights('Weights/' + session_name + '/' +
            str(segmentation_network.model_name) + '/Iteration_' + str(iteration) + '/')
    generator.load_weights('Weights/' + session_name + '/' +
            str(generator.model_name) + '/Iteration_' + str(iteration) + '/')
    discriminator.load_weights('Weights/' + session_name + '/' +
            str(discriminator.model_name) + '/Iteration_' + str(iteration) + '/')

    # Iterate over batches
    for batch_id, (batch_images_real, batch_labels) in enumerate(test_dataset):
        card = tf.data.experimental.cardinality(test_dataset)
        print(f'Batch: {batch_id+1}/{card}')

        # Get predictions
        batch_masks_logits = segmentation_network(batch_images_real)
        batch_size = batch_masks_logits.shape[0]

        batch_images_fake, z_k, k = generator(batch_images_real, batch_masks_logits,
                update_generator=True, training=False)

        z_k_hat = z_k

        d_logits_real = discriminator(batch_images_real, training=False)
        d_logits_fake = discriminator(batch_images_fake, training=False)

        print(f'd_logits_real: {d_logits_real}')
        print(f'd_logits_fake: {d_logits_fake}')

        g_loss_d, g_loss_i = loss.get_g_loss(d_logits_fake, z_k, z_k_hat)
        d_loss_r, d_loss_f = loss.get_d_loss(d_logits_real, d_logits_fake)
        print(f'Generator loss (discriminator): {g_loss_d}')
        print(f'Generator loss (information): {g_loss_i}')
        print(f'Discriminator loss (real): {d_loss_r}')
        print(f'Discriminator loss (fake): {d_loss_f}')

        for i, (image_real, mask_logits, image_fake) in enumerate(zip(
            batch_images_real, batch_masks_logits, batch_images_fake)):

            fig, ax = plt.subplots(1, 3)

            ax[0].set_title('Image')
            image = image_real.numpy() / (np.max(image_real) - \
                    np.min(image_real))
            image -= np.min(image)

            # Fake image with redrawn background
            image_fake_bg = image_fake.numpy()
            image_fake_bg -= np.min(image_fake_bg)
            image_fake_bg /= (np.max(image_fake_bg) - np.min(image_fake_bg))
            ax[0].imshow(image)
            ax[1].set_title('Prediction')
            ax[1].imshow(mask_logits.numpy()[:, :, 1], cmap='gray')
            ax[2].set_title('Fake Background')
            ax[2].imshow(image_fake_bg)
            plt.show()

