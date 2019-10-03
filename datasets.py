import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

######################
# Dataset Base Class #
######################


class Dataset:

    def __init__(self, root, image_dir, label_dir, path_file, split_file):
        self.root = root  # relative path to root directory of the dataset
        self.image_dir = image_dir  # sub-directory containing images
        self.label_dir = label_dir  # sub-directory containing segmentation masks
        self.path_file = path_file  # txt-file containing paths to each image and corresponding label
        # relative to image_dir and label_dir
        self.split_file = split_file  # txt-file containing split information for each image

        self.paths = self.read_file(os.path.join(self.root, self.path_file), file_type='path')  # list containing paths
        # to each image and corresponding label relative to image_dir and label_dir
        self.split_keys = self.read_file(os.path.join(self.root, self.split_file), file_type='split')  # list containing
        # the split key for each image

        self.split_key = {'training': 0, 'validation': 1, 'test': 2}  # available split options

        assert (len(self.paths) == len(self.split_keys))  # make sure split information available for each image

    def summary(self):
        print('##############################################')
        print('Dataset Summary: ')
        print('----------------------------------------------')
        print('Image Directory: {:s}'.format(os.path.join(self.root, self.image_dir)))
        print('Label Directory: {:s}'.format(os.path.join(self.root, self.label_dir)))
        print('Path File: {:s}'.format(os.path.join(self.root, self.path_file)))
        print('Split File: {:s}'.format(os.path.join(self.root, self.split_file)))
        print('##############################################')

    @staticmethod
    def read_file(filename, file_type):

        assert (file_type == 'path' or 'split')  # check for correct type of file

        items = []
        with open(filename, 'r') as file:
            for line in file:
                if file_type == 'path':
                    item = line.split(' ')[1].lstrip().rstrip()
                else:
                    item = int(line.split(' ')[1].lstrip().rstrip())  # convert split key to int
                items.append(item)

        file.close()
        return items

    @staticmethod
    def transform(image_path, label_path):
        """
        Process image_path and label_path
        :param image_path: relative path to image file
        :param label_path: relative path to label file
        :return: transformed input image (tf.tensor, shape: [128, 128, 3]),
                 transformed label (tf.tensor, shape: [128, 128, 1])
        """

        # load image and label as tf.tensor
        image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
        label = tf.image.decode_jpeg(tf.io.read_file(label_path), channels=1)

        # resize images to match required input dimensions
        image = tf.image.resize(image, size=(128, 128), preserve_aspect_ratio=False)
        label = tf.image.resize(label, size=(128, 128), preserve_aspect_ratio=False)

        # standardize image
        image = tf.image.per_image_standardization(image)

        # binarize and create one-hot label
        object_threshold = 0.9*tf.reduce_max(label)  # threshold for minimum intensity of object in non-binary label
        label = tf.cast(tf.where(label >= object_threshold, 1, 0), tf.uint8)[:, :, 0]
        label = tf.one_hot(label, depth=2)
        return image, label

    def get_split(self, split, size=None, batch_size=25, shuffle=False):
        """
        Get a split of the CUB dataset with given batch_size.
        :param split: desired split of the CUB dataset. Options: 'training', 'validation', 'test'
        :param size: number of samples in the dataset. Default: 'None' -> whole dataset
        :param batch_size: desired batch_size
        :return: tf.data.Dataset object containing the desired split of the CUB dataset
        """

        assert (split in self.split_key)  # make sure desired split is one of the options

        # get relative paths to images and labels of desired split | convert python list to tf.tensor
        split_image_paths = tf.convert_to_tensor([os.path.join(self.root, self.image_dir, ('image_' + path))
                                                  for i, path in enumerate(self.paths) if
                                                  self.split_key[split] == self.split_keys[i]])
        split_label_paths = tf.convert_to_tensor([os.path.join(self.root, self.label_dir, ('label_' + path))
                                                  for i, path in enumerate(self.paths) if
                                                  self.split_key[split] == self.split_keys[i]])

        # make sure a label exists for each image in split
        assert (split_image_paths.shape[0] == split_label_paths.shape[0])

        # randomly sample images from dataset to get desired size
        if size is not None:
            subset_indices = tf.random.uniform([size, 1], 0, split_image_paths.shape[0], dtype=tf.dtypes.int64)
            split_image_paths = tf.gather_nd(split_image_paths, subset_indices)
            split_label_paths = tf.gather_nd(split_label_paths, subset_indices)

        # create tf.data.dataset objects for images and labels | zip datasets to create (image, label) dataset
        split_image_ds = tf.data.Dataset.from_tensor_slices(split_image_paths)
        split_label_ds = tf.data.Dataset.from_tensor_slices(split_label_paths)
        split_ds = tf.data.Dataset.zip((split_image_ds, split_label_ds))

        # apply transform function to each tuple (image_path, label_path) in dataset
        split_ds = split_ds.map(self.transform)

        # set dataset parameters
        split_ds = split_ds.batch(batch_size=batch_size, drop_remainder=False)  # set batch size

        # set number of repetitions to 1
        split_ds = split_ds.repeat(1)

        # enable shuffling
        if shuffle:
            split_ds = split_ds.shuffle(buffer_size=tf.cast(tf.math.ceil(
                tf.convert_to_tensor(len(split_image_paths) / batch_size)), tf.int64))

        return split_ds


#########
# Birds #
#########


class BirdDataset(Dataset):

    def __init__(self, root='Datasets/Birds/', image_dir='images/', label_dir='labels/', path_file='paths.txt',
                        split_file='train_val_test_split.txt'):
        super().__init__(root, image_dir, label_dir, path_file, split_file)
        self.type = 'Bird'
        self.n_classes = 2

    @staticmethod
    def read_file(filename, file_type):

        assert (file_type == 'path' or 'split')  # check for correct type of file

        items = []
        with open(filename, 'r') as file:
            for line in file:
                if file_type == 'path':
                    item = line.split(' ')[1].lstrip().rstrip()[:-4]  # strip of '.jpg' at the end of the image path
                else:
                    item = int(line.split(' ')[1].lstrip().rstrip())  # convert split key to int
                items.append(item)

        file.close()
        return items

    def get_split(self, split, size=None, batch_size=25, shuffle=False):
        """
        Get a split of the CUB dataset with given batch_size.
        :param split: desired split of the CUB dataset. Options: 'training', 'validation', 'test'
        :param batch_size: desired batch_size
        :param size: number of samples in the dataset. Default: 'None' -> whole dataset
        :param shuffle: enable shuffling or not
        :return: tf.data.Dataset object containing the desired split of the CUB dataset
        """

        assert (split in self.split_key)  # make sure desired split is one of the options

        # get relative paths to images and labels of desired split | convert python list to tf.tensor
        split_image_paths = tf.convert_to_tensor([os.path.join(self.root, self.image_dir, (path + '.jpg'))
                                                  for i, path in enumerate(self.paths) if
                                                  self.split_key[split] == self.split_keys[i]])
        split_label_paths = tf.convert_to_tensor([os.path.join(self.root, self.label_dir, (path + '.png'))
                                                  for i, path in enumerate(self.paths) if
                                                  self.split_key[split] == self.split_keys[i]])

        # make sure a label exists for each image in split
        assert (split_image_paths.shape[0] == split_label_paths.shape[0])

        # randomly sample images from dataset to get desired size
        if size is not None:
            subset_indices = tf.random.uniform([size, 1], 0, split_image_paths.shape[0], dtype=tf.dtypes.int64)
            split_image_paths = tf.gather_nd(split_image_paths, subset_indices)
            split_label_paths = tf.gather_nd(split_label_paths, subset_indices)

        # create tf.data.dataset objects for images and labels | zip datasets to create (image, label) dataset
        split_image_ds = tf.data.Dataset.from_tensor_slices(split_image_paths)
        split_label_ds = tf.data.Dataset.from_tensor_slices(split_label_paths)
        split_ds = tf.data.Dataset.zip((split_image_ds, split_label_ds))

        # apply transform function to each tuple (image_path, label_path) in dataset
        split_ds = split_ds.map(self.transform)

        # set dataset parameters
        split_ds = split_ds.batch(batch_size=batch_size, drop_remainder=False)  # set batch size

        # set number of repetitions to 1
        split_ds = split_ds.repeat(1)

        # enable shuffling
        if shuffle:
            split_ds = split_ds.shuffle(buffer_size=tf.cast(tf.math.ceil(
                tf.convert_to_tensor(len(split_image_paths)/batch_size)), tf.int64))

        return split_ds


###########
# Flowers #
###########

class FlowerDataset(Dataset):

    def __init__(self, root='Datasets/Flowers/', image_dir='images/', label_dir='labels/', path_file='paths.txt',
                        split_file='train_val_test_split.txt'):
        super().__init__(root, image_dir, label_dir, path_file, split_file)
        self.type = 'Flower'
        self.n_classes = 2

    @staticmethod
    def transform(image_path, label_path):
        """
        Process image_path and label_path
        :param image_path: relative path to image file
        :param label_path: relative path to label file
        :return: transformed input image (tf.tensor, shape: [128, 128, 3]),
                 transformed label (tf.tensor, shape: [128, 128, 1])
        """

        # load image and label as tf.tensor
        image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
        label = tf.image.decode_png(tf.io.read_file(label_path), channels=1)

        # resize images to match required input dimensions
        image = tf.image.resize(image, size=(128, 128), preserve_aspect_ratio=False)
        label = tf.image.resize(label, size=(128, 128), preserve_aspect_ratio=False)

        # standardize image
        image = tf.image.per_image_standardization(image)

        # binarize and get one-hot label
        background_color = 29
        label = tf.cast(tf.where(tf.logical_or(label <= 0.9*background_color, label >= 1.1*background_color), 1, 0),
                        tf.uint8)[:, :, 0]
        label = tf.one_hot(label, depth=2)

        return image, label


#########
# Faces #
#########

class FaceDataset(Dataset):

    def __init__(self, root='Datasets/Faces/', image_dir='images/', label_dir='labels/', path_file='paths.txt',
                        split_file='train_val_test_split.txt'):
        super().__init__(root, image_dir, label_dir, path_file, split_file)
        self.type = 'Face'
        self.n_classes = 2

    def get_split(self, split, size=None, batch_size=25, shuffle=False):
        """
        Get a split of the CUB dataset with given batch_size.
        :param split: desired split of the CUB dataset. Options: 'training', 'validation', 'test'
        :param size: number of samples in the dataset. Default: 'None' -> whole dataset
        :param batch_size: desired batch_size
        :return: tf.data.Dataset object containing the desired split of the CUB dataset
        """

        assert (split in self.split_key)  # make sure desired split is one of the options

        # get relative paths to images and labels of desired split | convert python list to tf.tensor
        split_image_paths = tf.convert_to_tensor([os.path.join(self.root, self.image_dir, (path[:-9] + '/' + path))
                                                  for i, path in enumerate(self.paths) if
                                                  self.split_key[split] == self.split_keys[i]])

        if split == 'training':
            split_label_paths = tf.convert_to_tensor([os.path.join(self.root, self.image_dir, (path[:-9] + '/' + path))
                                                      for i, path in enumerate(self.paths) if
                                                      self.split_key[split] == self.split_keys[i]])

        else:
            split_label_paths = tf.convert_to_tensor([os.path.join(self.root, self.label_dir, path)
                                                  for i, path in enumerate(self.paths) if
                                                  self.split_key[split] == self.split_keys[i]])

        # make sure a label exists for each image in split
        assert (split_image_paths.shape[0] == split_label_paths.shape[0])

        # randomly sample images from dataset to get desired size
        if size is not None:
            subset_indices = tf.random.uniform([size, 1], 0, split_image_paths.shape[0], dtype=tf.dtypes.int64)
            split_image_paths = tf.gather_nd(split_image_paths, subset_indices)
            split_label_paths = tf.gather_nd(split_label_paths, subset_indices)

        # create tf.data.dataset objects for images and labels | zip datasets to create (image, label) dataset
        split_image_ds = tf.data.Dataset.from_tensor_slices(split_image_paths)
        split_label_ds = tf.data.Dataset.from_tensor_slices(split_label_paths)
        split_ds = tf.data.Dataset.zip((split_image_ds, split_label_ds))

        # apply transform function to each tuple (image_path, label_path) in dataset
        split_ds = split_ds.map(self.transform)

        # set dataset parameters
        split_ds = split_ds.batch(batch_size=batch_size, drop_remainder=False)  # set batch size

        # set number of repetitions to 1
        split_ds = split_ds.repeat(1)

        # enable shuffling
        if shuffle:
            split_ds = split_ds.shuffle(buffer_size=tf.cast(tf.math.ceil(
                tf.convert_to_tensor(len(split_image_paths)/batch_size)), tf.int64))

        return split_ds

    @staticmethod
    def transform(image_path, label_path):
        """
        Process image_path and label_path
        :param image_path: relative path to image file
        :param label_path: relative path to label file
        :return: transformed input image (tf.tensor, shape: [128, 128, 3]),
                 transformed label (tf.tensor, shape: [128, 128, 1])
        """

        # load image and label as tf.tensor
        image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
        label = tf.image.decode_jpeg(tf.io.read_file(label_path), channels=1)

        # resize images to match required input dimensions
        image = tf.image.resize(image, size=(128, 128), preserve_aspect_ratio=False)
        label = tf.image.resize(label, size=(128, 128), preserve_aspect_ratio=False)

        # standardize image
        image = tf.image.per_image_standardization(image)

        # binarize and get one-hot label
        object_threshold = 40  # threshold for minimum intensity of object in non-binary label
        label = tf.cast(tf.where(label > object_threshold, 1, 0), tf.uint8)[:, :, 0]
        label = tf.one_hot(label, depth=2)

        return image, label


if __name__ == '__main__':

    # Birds
    Birds = BirdDataset(root='Datasets/Birds/', image_dir='images/', label_dir='labels/', path_file='paths.txt',
                     split_file='train_val_test_split.txt')
    Birds.summary()
    birds_training = Birds.get_split(split='training', size=400)
    birds_validation = Birds.get_split(split='validation')
    birds_test = Birds.get_split(split='test')

    # Flowers
    Flowers = FlowerDataset(root='Datasets/Flowers/', image_dir='images/', label_dir='labels/', path_file='paths.txt',
                             split_file='train_val_test_split.txt')
    Flowers.summary()
    flowers_training = Flowers.get_split(split='training')
    flowers_validation = Flowers.get_split(split='validation')
    flowers_test = Flowers.get_split(split='test')

    # Faces
    Faces = FaceDataset(root='Datasets/Faces/', image_dir='images/', label_dir='labels/', path_file='paths.txt',
                             split_file='train_val_test_split.txt')
    Faces.summary()
    faces_training = Faces.get_split(split='training')
    faces_validation = Faces.get_split(split='validation', size=400, shuffle=True)
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
