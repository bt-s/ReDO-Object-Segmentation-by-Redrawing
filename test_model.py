import tensorflow as tf
from networks import MaskGenerator
from datasets import BirdDataset, FlowerDataset, FaceDataset
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Mean, Accuracy, MeanIoU
from train_utils import SupervisedLoss

if __name__ == '__main__':

    # create datasets
    dataset = BirdDataset(root='Datasets/Birds/', image_dir='images/', label_dir='labels/', path_file='paths.txt',
                        split_file='train_val_test_split.txt')
    test_dataset = dataset.get_split('test', batch_size=25)

    # create model and load weights
    model = MaskGenerator(n_classes=dataset.n_classes)
    epoch = 7
    model.load_weights('Weights/' + str(dataset.type) + '_Epoch_' + str(epoch))

    # loss function
    loss = SupervisedLoss()

    # define metrics dictionary
    metrics = {'test_loss': Mean(), 'test_accuracy': Mean(), 'test_IoU': Mean()}
    # batch metrics
    accuracy = Accuracy()
    iou = MeanIoU(model.n_classes)

    # iterate over batches
    for batch_id, (batch_images, batch_labels) in enumerate(test_dataset):

        # get predictions
        batch_predictions = model(batch_images)
        # compute loss for current batch
        batch_loss = loss(batch_predictions, batch_labels)

        # update respective metric with computed loss
        metrics['test_loss'](batch_loss)
        batch_accuracy = accuracy(tf.argmax(batch_predictions, axis=3), tf.argmax(batch_labels, axis=3))
        metrics['test_accuracy'](batch_accuracy)
        batch_iou = iou(tf.argmax(batch_predictions, axis=3), tf.argmax(batch_labels, axis=3))
        metrics['test_IoU'](batch_iou)

        for image, label in zip(batch_predictions, batch_labels):
            fig, ax = plt.subplots(1, 2)
            ax[0].set_title('Prediction')
            ax[0].imshow(tf.math.argmax(image, axis=2).numpy(), cmap='gray')
            ax[0].set_title('Label')
            ax[1].imshow(label.numpy()[:, :, 0], cmap='gray')
            plt.show()

    # print summary at the end of epoch
    test_summary = 'Loss: {:.6f}, Accuracy: {:.6f}, IoU: {:.6f}'
    print(test_summary.format(metrics['test_loss'].result(), metrics['test_accuracy'].result(),
                              metrics['test_IoU'].result()))


