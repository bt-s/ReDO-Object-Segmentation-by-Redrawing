import tensorflow as tf
from segmentation_network import MaskGenerator
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
    test_dataset = dataset.get_split('test', batch_size=25)

    # initializer
    init_gain = 1.0

    # create model and load weights
    model = MaskGenerator(n_classes=dataset.n_classes, init_gain=init_gain)
    model.set_name('Segmentation_Network')
    epoch = 10
    model.load_weights('Weights/' + session_name + '/' + str(model.model_name) + '/Epoch_' + str(epoch) + '/')

    # loss function
    loss = SupervisedLoss()

    # define metrics dictionary
    metrics = {'test_loss': Mean(), 'test_accuracy': Mean(), 'test_IoU': Mean()}

    # iterate over batches
    for batch_id, (batch_images, batch_labels) in enumerate(test_dataset):

        # print progress
        print('Batch: {:d}/{:d}'.format(batch_id + 1, tf.data.experimental.cardinality(test_dataset)))

        # get predictions
        batch_predictions = model(batch_images)
        # compute loss for current batch
        batch_loss = loss(batch_predictions, batch_labels)

        # update respective metric with computed loss
        metrics['test_loss'](batch_loss)
        batch_accuracy = train_utils.compute_accuracy(batch_predictions, batch_labels)
        metrics['test_accuracy'](batch_accuracy)
        batch_iou = train_utils.compute_IoU(batch_predictions, batch_labels)
        metrics['test_IoU'](batch_iou)

        for image, prediction, label in zip(batch_images, batch_predictions, batch_labels):
            fig, ax = plt.subplots(1, 3)
            ax[0].set_title('Image')
            image = image.numpy() / (np.max(image) - np.min(image))
            image -= np.min(image)
            ax[0].imshow(image)
            ax[1].set_title('Prediction')
            ax[1].imshow(tf.keras.layers.Softmax(axis=2)(prediction).numpy()[:, :, 0], cmap='gray')
            ax[2].set_title('Label')
            ax[2].imshow(label.numpy()[:, :, 1], cmap='gray')
            plt.show()

    # print summary at the end of epoch
    test_summary = 'Loss: {:.6f}, Accuracy: {:.6f}, IoU: {:.6f}'
    print(test_summary.format(metrics['test_loss'].result(), metrics['test_accuracy'].result(),
                              metrics['test_IoU'].result()))


