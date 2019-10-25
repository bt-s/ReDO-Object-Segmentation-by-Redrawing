#!/usr/bin/python3

"""evaluate_masks.py - Script to test mask generator

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chmielewski-Anders, Mats Steinweg & Bas Straathof"


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.metrics import Mean, Accuracy, MeanIoU

from datasets import BirdDataset, FlowerDataset, FaceDataset
from segmentation_network import SegmentationNetwork
from train_utils import *


if __name__ == '__main__':
    session_name = 'Unsupervised_Flowers'

    # Create datasets
    dataset = FlowerDataset()
    test_dataset = dataset.get_split('test', batch_size=25)

    # Create model and load weights
    model = SegmentationNetwork(n_classes=dataset.n_classes,
            init_gain=0.0, weight_decay=1e-4)

    iteration = 2500
    model.load_weights((f'Weights/{session_name}/{str(model.model_name)}/' \
            f'Iteration_{iteration}/'))

    # Loss function
    loss = SupervisedLoss()

    # Define metrics dictionary
    metrics = {'test_loss': Mean(), 'test_accuracy': Mean(), 'test_IoU': Mean()}

    # Iterate over batches
    for batch_id, (batch_images, batch_labels) in enumerate(test_dataset):

        card = tf.data.experimental.cardinality(test_dataset)
        print(f'Batch: {batch_id+1}/{card}')

        # Get predictions
        batch_predictions = model(batch_images)
        # Compute loss for current batch
        batch_loss = loss(batch_predictions, batch_labels)

        # Update respective metric with computed loss
        metrics['test_loss'](batch_loss)
        batch_accuracy = compute_accuracy(batch_predictions, batch_labels)
        metrics['test_accuracy'](batch_accuracy)
        batch_iou = compute_IoU(batch_predictions, batch_labels)
        metrics['test_IoU'](batch_iou)

        for image, prediction, label in zip(batch_images, batch_predictions,
                batch_labels):
            fig, ax = plt.subplots(1, 3)
            ax[0].set_title('Image')
            image = image.numpy() / (np.max(image) - np.min(image))
            image -= np.min(image)
            ax[0].imshow(image)
            ax[1].set_title('Prediction')
            ax[1].imshow(prediction.numpy()[:, :, 1], cmap='gray', vmin=0.0, vmax=1.0)
            ax[2].set_title('Label')
            ax[2].imshow(label.numpy()[:, :, 1], cmap='gray')
            plt.show()

    # Print summary at the end of epoch
    test_summary = 'Loss: {:.6f}, Accuracy: {:.6f}, IoU: {:.6f}'
    print(test_summary.format(metrics['test_loss'].result(),
        metrics['test_accuracy'].result(), metrics['test_IoU'].result()))


