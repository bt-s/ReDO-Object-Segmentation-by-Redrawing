#!/usr/bin/python3

"""datasets.py - Handling of supported datasets. For getting the datasets see
                 `get_datasets.py`

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chmielewski-Anders, Mats Steinweg & Bas Straathof"


import matplotlib.pyplot as plt
import numpy as np

from datasets import BirdDataset, FlowerDataset, FaceDataset


if __name__ == '__main__':

    # Birds
    Birds = BirdDataset(root='../Datasets/Birds/', image_dir='images/',
                        label_dir='labels/', path_file='paths.txt',
                        split_file='train_val_test_split.txt')
    Birds.summary()
    birds_training = Birds.get_split(split='training', size=400)
    birds_validation = Birds.get_split(split='validation')
    birds_test = Birds.get_split(split='test')

    # Flowers
    Flowers = FlowerDataset(root='../Datasets/Flowers/', image_dir='images/',
                            label_dir='labels/', path_file='paths.txt',
                            split_file='train_val_test_split.txt')
    Flowers.summary()
    flowers_training = Flowers.get_split(split='training')
    flowers_validation = Flowers.get_split(split='validation')
    flowers_test = Flowers.get_split(split='test')

    # Faces
    exit(0)
    # Not supported at the moment
    Faces = FaceDataset(root='../Datasets/Faces/', image_dir='images/',
                        label_dir='labels/', path_file='paths.txt',
                        split_file='train_val_test_split.txt')

    Faces.summary()
    faces_training = Faces.get_split(split='training')
    faces_validation = Faces.get_split(split='validation', size=400,
                                       shuffle=True)
    faces_test = Faces.get_split(split='test')

    for idx, (batch_images, batch_labels) in enumerate(flowers_training):
        for image, label in zip(batch_images, batch_labels):
            fig, ax = plt.subplots(1, 2)
            image = image.numpy()
            image -= np.min(image)
            image /= (np.max(image) - np.min(image))
            ax[0].imshow(image)
            ax[1].imshow(label.numpy()[:, :, 1], cmap='gray')
            plt.show()
