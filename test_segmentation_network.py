#!/usr/bin/python3

"""datasets.py - Handling of supported datasets. For getting the datasets see
                 `get_datasets.py`

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chmielewski-Anders, Mats Steinweg & Bas Straathof"


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Softmax

from segmentation_network import SegmentationNetwork

if __name__ == '__main__':
    # Prepare exemplary image batch
    input_path_1 = '../Datasets/Flowers/images/image_00023.jpg'
    input_path_2 = '../Datasets/Flowers/images/image_00081.jpg'
    image_1 = tf.image.decode_jpeg(tf.io.read_file(input_path_1))
    image_1 = tf.image.resize(image_1, (128, 128), preserve_aspect_ratio=False)
    image_1 = tf.expand_dims(tf.image.per_image_standardization(image_1), 0)
    image_2 = tf.image.decode_jpeg(tf.io.read_file(input_path_2))
    image_2 = tf.image.resize(tf.image.per_image_standardization(image_2),
            (128, 128), preserve_aspect_ratio=False)
    image_2 = tf.expand_dims(image_2, 0)
    image_batch = tf.concat((image_1, image_2), 0)

    # Create network object
    f = SegmentationNetwork(n_classes=2, init_gain=0.8, weight_decay=1e-4)

    # Forward pass
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

    # Plot images and masks
    ax[0, 0].imshow(image_1)
    ax[1, 0].imshow(image_2)
    ax[0, 1].imshow(mask_1, cmap='gray')
    ax[1, 1].imshow(mask_2, cmap='gray')
    plt.show()

