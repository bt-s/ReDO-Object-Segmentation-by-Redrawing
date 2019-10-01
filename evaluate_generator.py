import tensorflow as tf
from segmentation_network import MaskGenerator
from generator import Generator
from datasets import BirdDataset, FlowerDataset, FaceDataset
from tensorflow.keras.metrics import Mean, Accuracy, MeanIoU
from train_utils import SupervisedLoss
import train_utils
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    session_name = 'Unsupervised_Flowers'

    # create datasets
    dataset = FlowerDataset()
    test_dataset = dataset.get_split('training', batch_size=25)

    # initializer
    init_gain = 1.0

    # create model and load weights
    model = MaskGenerator(n_classes=dataset.n_classes, init_gain=init_gain)
    model.set_name('Segmentation_Network')
    generator = Generator(init_gain=init_gain, input_dim=32, base_channels=32)
    generator.set_region(k=0)
    generator.set_name('Generator_0')
    epoch = 59
    model.load_weights('Weights/' + session_name + '/' + str(model.model_name) + '/Epoch_' + str(epoch) + '/')
    generator.load_weights('Weights/' + session_name + '/' + str(model.model_name) + '/Epoch_' + str(epoch) + '/')

    # iterate over batches
    for batch_id, (batch_images, batch_labels) in enumerate(test_dataset):

        # print progress
        print('Batch: {:d}/{:d}'.format(batch_id + 1, tf.data.experimental.cardinality(test_dataset)))

        # get predictions
        batch_predictions = model(batch_images)

        generated_images, z_k = generator(batch_images, batch_predictions, training=False)

        for image, prediction, redrawn_image in zip(batch_images, batch_predictions, generated_images):
            fig, ax = plt.subplots(1, 3)
            ax[0].set_title('Image')
            image = image.numpy() / (np.max(image) - np.min(image))
            image -= np.min(image)
            ax[0].imshow(image)
            ax[1].set_title('Prediction')
            ax[1].imshow(tf.keras.layers.Softmax(axis=2)(prediction).numpy()[:, :, 0], cmap='gray')
            ax[2].set_title('Composed Image')
            ax[2].imshow(redrawn_image.numpy())
            plt.show()

