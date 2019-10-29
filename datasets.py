#!/usr/bin/python3

"""datasets.py - Handling of supported datasets. For getting the datasets see
                 `get_datasets.py`

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chiemelewski-Anders, Mats Steinweg & Bas Straathof"


import tensorflow as tf
import os

from typing import Union, List, Tuple


class Dataset:
    """Dataset base class used as superclass for supported dataset subclasses"""
    SPLIT_KEYS = {'training': 0, 'validation': 1, 'test': 2}

    def __init__(self, root: str, image_dir: str, label_dir: str,
                 path_file: str, split_file: str):
        """Class constructor
        Args:
            root: relative path to root directory of the dataset
            image_dir: sub-directory containing images
            label_dir: sub-directory containing segmentation masks
            path_file: txt-file containing paths to each image and corresponding
                       label
            split_file: relative to image_dir and label_dir txt-file
                        containing split information for each image
        """
        self.root = root
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.path_file = path_file
        self.split_file = split_file

        # List containing paths to each image and corresponding label
        # relative to image_dir and label_dir
        self.paths = self.read_file(os.path.join(
            self.root, self.path_file), file_type='path')

        # List containing the split key for each image
        self.split_keys = self.read_file(
            os.path.join(self.root, self.split_file), file_type='split')

        # Make sure split information available for each image
        assert (len(self.paths) == len(self.split_keys))


    def summary(self):
        """Prints a summary of the data set"""
        print('##############################################')
        print('Dataset Summary: ')
        print('----------------------------------------------')
        print('Image Directory: {:s}'.format(os.path.join(self.root,
                                                          self.image_dir)))
        print('Label Directory: {:s}'.format(os.path.join(self.root,
                                                          self.label_dir)))
        print('Path File: {:s}'.format(os.path.join(self.root, self.path_file)))
        print('Split File: {:s}'.format(os.path.join(self.root,
                                                     self.split_file)))

        print('##############################################')


    @staticmethod
    def read_file(filename: str, file_type: str) -> Union[List[str], List[int]]:
        """Read either a path.txt file or split file

        Args:
            filename: the name of the file to be read
            file_type: one of path, split

        Returns:
            A list of paths (strings) or splits (ints) corresponding to
            train/val/test

        Raises:
            ValueError: if `file_type` is not one of the correct values
        """
        # check for correct type of file
        # if not (file_type == 'path' or 'split'):
        #   raise ValueError(f'{file_type} not one of path, split')

        items = []
        with open(filename, 'r') as file:
            for line in file:
                if file_type == 'path':
                    item = line.split(' ')[1].lstrip().rstrip()
                else:
                    # convert split key to int
                    item = int(line.split(' ')[1].lstrip().rstrip())
                items.append(item)

        file.close()

        return items


    @staticmethod
    def transform(image_path: str, label_path: str) -> Tuple[tf.Tensor,
            tf.Tensor]:
        """Process image_path and label_path

        Args:
            image_path: relative path to image file
            label_path: relative path to label file

        Returns:
            transformed input image (tf.tensor, shape: [128, 128, 3]),
            transformed label (tf.tensor, shape: [128, 128, 1])
        """
        # Load image and label as tf.tensor
        image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
        label = tf.image.decode_jpeg(tf.io.read_file(label_path), channels=1)

        # Resize images to match required input dimensions
        image = tf.image.resize(image, size=(128, 128),
                preserve_aspect_ratio=False)
        label = tf.image.resize(label, size=(128, 128),
                preserve_aspect_ratio=False)

        # Center image
        image = (image / 255.0) * 2 - 1

        # Binarize and create one-hot label
        # Threshold for minimum intensity of object in non-binary label
        object_threshold = 0.9*tf.reduce_max(label)
        label = tf.cast(tf.where(label >= object_threshold, 1, 0),
                tf.uint8)[:, :, 0]

        label = tf.one_hot(label, depth=2)

        return image, label


    def get_split(self, split: str, size: int=None, batch_size: int=25,
            shuffle: bool=False) -> tf.data.Dataset:
        """Get a split of the CUB dataset with given batch_size.

        Args:
            split: desired split of the CUB dataset. Options: 'training',
                   'validation', 'test'
            size: number of samples in the dataset. Default: 'None' -> whole
                  dataset
            batch_size: desired batch_size
            shuffle: True to shuffle the dataset

        Returns:
            tf.data.Dataset object containing the desired split of the CUB
            dataset

        Raises:
            ValueError if `split` not in SPLIT_KEYS
        """
        # if split not in Dataset.SPLIT_KEYS:
        #    raise ValueError(f'{split} not one of {Dataset.SPLIT_KEYS}')

        # Get relative paths to images and labels of desired split
        # convert python list to tf.tensor
        split_image_paths = []
        split_label_paths = []
        for i, path in enumerate(self.paths):
            if self.split_keys[i] == Dataset.SPLIT_KEYS[split]:
                split_image_paths.append(os.path.join(self.root, self.image_dir,
                    ('image_' + path)))
                split_label_paths.append(os.path.join(self.root, self.label_dir,
                    ('label_' + path)))

        split_image_paths = tf.convert_to_tensor(split_image_paths)
        split_label_paths = tf.convert_to_tensor(split_label_paths)

        # Randomly sample images from dataset to get desired size
        if size is not None:
            subset_indices = tf.random.uniform([size, 1], 0,
                    split_image_paths.shape[0], dtype=tf.dtypes.int64)

            split_image_paths = tf.gather_nd(split_image_paths, subset_indices)
            split_label_paths = tf.gather_nd(split_label_paths, subset_indices)
        else:
            indices = tf.range(0, split_image_paths.shape[0],
                    dtype=tf.dtypes.int64)
            shuffled_indices = tf.random.shuffle(indices)
            split_image_paths = tf.gather(split_image_paths, shuffled_indices)
            split_label_paths = tf.gather(split_label_paths, shuffled_indices)

        # Create tf.data.dataset objects for images and labels
        # Zip datasets to create (image, label) dataset
        split_image_ds = tf.data.Dataset.from_tensor_slices(split_image_paths)
        split_label_ds = tf.data.Dataset.from_tensor_slices(split_label_paths)
        split_ds = tf.data.Dataset.zip((split_image_ds, split_label_ds))

        # Apply transform function to each tuple (image_path, label_path)
        # in dataset
        split_ds = split_ds.map(self.transform)

        # Set dataset parameters, batch size
        split_ds = split_ds.batch(batch_size=batch_size, drop_remainder=True)

        # Set number of repetitions to 1
        split_ds = split_ds.repeat(1)

        # Enable shuffling
        if shuffle:
            split_ds = split_ds.shuffle(buffer_size=tf.cast(
                tf.math.ceil(tf.convert_to_tensor( len(split_image_paths) / \
                        batch_size)), tf.int64))

        return split_ds


class BirdDataset(Dataset):
    """Class to support the birds dataset"""
    def __init__(self, root: str='Datasets/Birds/', image_dir: str='images/',
                 label_dir: str='labels/', path_file: str='paths.txt',
                 split_file: str='train_val_test_split.txt'):
        """See `Dataset.__init__`"""
        super().__init__(root, image_dir, label_dir, path_file, split_file)
        self.type = 'Bird'
        self.n_classes = 2


    @staticmethod
    def read_file(filename: str, file_type: str):
        """See `Dataset.read_file` but with specific formatting for the
        dataset"""

        # Check for correct type of file
        # if not (file_type == 'path' or 'split'):
        #    raise ValueError(f'{file_type} not one of path, split')

        items = []
        with open(filename, 'r') as file:
            for line in file:
                if file_type == 'path':
                    # Strip of '.jpg' at the end of the image path
                    item = line.split(' ')[1].lstrip().rstrip()[:-4]
                else:
                    # Convert split key to int
                    item = int(line.split(' ')[1].lstrip().rstrip())
                items.append(item)

        file.close()

        return items

    def get_split(self, split: str, size: int=None, batch_size: int=25,
            shuffle: bool=False) -> tf.data.Dataset:
        """See `Dataset.get_split`"""
        # if split not in Dataset.SPLIT_KEYS:
        #    raise ValueError(f'{split} not one of {Dataset.SPLIT_KEYS}')

        # Get relative paths to images and labels of desired split
        # convert python list to tf.tensor
        split_image_paths = []
        split_label_paths = []

        for i, path in enumerate(self.paths):
            if Dataset.SPLIT_KEYS[split] == self.split_keys[i]:
                split_image_paths.append(os.path.join(self.root,
                    self.image_dir, (path + '.jpg')))
                split_label_paths.append(os.path.join(self.root,
                    self.label_dir, (path + '.png')))

        split_image_paths = tf.convert_to_tensor(split_image_paths)
        split_label_paths = tf.convert_to_tensor(split_label_paths)

        # Randomly sample images from dataset to get desired size
        if size is not None:
            subset_indices = tf.random.uniform([size, 1], 0,
                    split_image_paths.shape[0], dtype=tf.dtypes.int64)

            split_image_paths = tf.gather_nd(split_image_paths, subset_indices)
            split_label_paths = tf.gather_nd(split_label_paths, subset_indices)
        else:
            indices = tf.range(0, split_image_paths.shape[0],
                    dtype=tf.dtypes.int64)
            shuffled_indices = tf.random.shuffle(indices)
            split_image_paths = tf.gather(split_image_paths, shuffled_indices)
            split_label_paths = tf.gather(split_label_paths, shuffled_indices)

        # Create tf.data.dataset objects for images and labels
        # Zip datasets to create (image, label) dataset
        split_image_ds = tf.data.Dataset.from_tensor_slices(split_image_paths)
        split_label_ds = tf.data.Dataset.from_tensor_slices(split_label_paths)
        split_ds = tf.data.Dataset.zip((split_image_ds, split_label_ds))

        # Apply transform function to each tuple (image_path, label_path)
        # in dataset
        split_ds = split_ds.map(self.transform)

        # Set dataset parameters, batch size
        split_ds = split_ds.batch(batch_size=batch_size, drop_remainder=True)

        # Set number of repetitions to 1
        split_ds = split_ds.repeat(1)

        # Enable shuffling
        if shuffle:
            split_ds = split_ds.shuffle(buffer_size=tf.cast(
                tf.math.ceil(tf.convert_to_tensor(len(split_image_paths) / \
                        batch_size)), tf.int64))

        return split_ds


class FlowerDataset(Dataset):
    """Subclass for the Flowers dataset"""
    def __init__(self, root: str='../../Datasets/Flowers/', image_dir:
            str='images/', label_dir: str='labels/', path_file: str='paths.txt',
            split_file: str='train_val_test_split.txt'):
        """See `Dataset.__init__`"""
        super().__init__(root, image_dir, label_dir, path_file, split_file)
        self.type = 'Flower'
        self.n_classes = 2


    @staticmethod
    def transform(image_path: str, label_path: str) -> Tuple[tf.Tensor,
            tf.Tensor]:
        """See `Dataset.transform`"""
        # Load image and label as tf.tensor
        image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
        label = tf.image.decode_png(tf.io.read_file(label_path), channels=1)

        # Resize images to match required input dimensions
        image = tf.image.resize(image, size=(128, 128),
                preserve_aspect_ratio=False)
        label = tf.image.resize(label, size=(128, 128),
                preserve_aspect_ratio=False)

        # Center image
        image = (image / 255.0) * 2 - 1

        # Binarize and get one-hot label
        background_color = 29
        label = tf.cast(tf.where(tf.logical_or(label <= 0.9 * background_color,
            label >= 1.1*background_color), 1, 0), tf.uint8)[:, :, 0]
        label = tf.one_hot(label, depth=2)

        return image, label


class FaceDataset(Dataset):
    """Subclass for the Face dataset"""
    def __init__(self, root: str='Datasets/Faces/', image_dir: str='images/',
            label_dir: str='labels/', path_file: str='paths.txt', split_file:
            str='train_val_test_split.txt'):
        """See `Dataset.__init__`"""
        super().__init__(root, image_dir, label_dir, path_file, split_file)
        self.type = 'Face'
        self.n_classes = 2


    def get_split(self, split: str, size: int=None, batch_size: int=25,
            shuffle: bool=False) -> tf.data.Dataset:
        """See `Dataset.get_split` differences specific to the dataset file
        organization"""
        # if split not in Dataset.SPLIT_KEYS:
        #    raise ValueError(f'{split} not one of {Dataset.SPLIT_KEYS}')

        # Get relative paths to images and labels of desired split
        # Convert python list to tf.tensor
        split_image_paths = []
        split_label_paths = []
        for i, path in enumerate(self.paths):
            if Dataset.SPLIT_KEYS[split] == self.split_keys[i]:
                split_image_paths.append(os.path.join(self.root,
                    self.image_dir, (path[:-9] + '/' + path)))
                if split == 'training' or split == 'validation':
                    split_label_paths.append(os.path.join(self.root,
                        self.image_dir, (path[:-9] + '/' + path)))
                else:
                    split_label_paths.append(os.path.join(self.root,
                        self.label_dir, path))

        split_image_paths = tf.convert_to_tensor(split_image_paths)
        split_label_paths = tf.convert_to_tensor(split_label_paths)

        if split_image_paths.shape[0] != split_label_paths.shape[0]:
            raise ValueError('split_image_paths shape != split_label_paths '
                             'shape')

        # Randomly sample images from dataset to get desired size
        if size is not None:
            subset_indices = tf.random.uniform([size, 1], 0,
                    split_image_paths.shape[0], dtype=tf.dtypes.int64)

            split_image_paths = tf.gather_nd(split_image_paths, subset_indices)
            split_label_paths = tf.gather_nd(split_label_paths, subset_indices)
        else:
            indices = tf.range(0, split_image_paths.shape[0],
                    dtype=tf.dtypes.int64)
            shuffled_indices = tf.random.shuffle(indices)
            split_image_paths = tf.gather(split_image_paths, shuffled_indices)
            split_label_paths = tf.gather(split_label_paths, shuffled_indices)

        # Create tf.data.dataset objects for images and labels
        # Zip datasets to create (image, label) dataset
        split_image_ds = tf.data.Dataset.from_tensor_slices(split_image_paths)
        split_label_ds = tf.data.Dataset.from_tensor_slices(split_label_paths)
        split_ds = tf.data.Dataset.zip((split_image_ds, split_label_ds))

        # Apply transform function to each tuple (image_path, label_path) in
        # dataset
        split_ds = split_ds.map(self.transform)

        # Set dataset parameters, batch size
        split_ds = split_ds.batch(batch_size=batch_size, drop_remainder=True)

        # Set number of repetitions to 1
        split_ds = split_ds.repeat(1)

        # Enable shuffling
        if shuffle:
            split_ds = split_ds.shuffle(buffer_size=tf.cast(tf.math.ceil(
                tf.convert_to_tensor( len(split_image_paths)/batch_size)),
                tf.int64))

        return split_ds


    @staticmethod
    def transform(image_path: str, label_path: str) -> Tuple[tf.Tensor,
            tf.Tensor]:
        """See `Dataset.transform` but specific to sizes of dataset"""
        # Load image and label as tf.tensor
        image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
        label = tf.image.decode_jpeg(tf.io.read_file(label_path), channels=1)

        # Resize images to match required input dimensions
        image = tf.image.resize(image, size=(128, 128),
                preserve_aspect_ratio=False)
        label = tf.image.resize(label, size=(128, 128),
                preserve_aspect_ratio=False)

        # Center image
        image = (image / 255.0) * 2 - 1

        # Binarize and get one-hot label
        # threshold for minimum intensity of object innon-binary label
        object_threshold = 40
        label = tf.cast(tf.where(label > object_threshold, 1, 0),
                tf.uint8)[:, :, 0]
        label = tf.one_hot(label, depth=2)

        return image, label

