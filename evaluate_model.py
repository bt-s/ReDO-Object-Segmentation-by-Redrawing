import tensorflow as tf
from networks import MaskGenerator
from datasets import BirdDataset, FlowerDataset, FaceDataset
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # create datasets
    Birds = BirdDataset(root='Datasets/Birds/', image_dir='images/', label_dir='labels/', path_file='paths.txt',
                        split_file='train_val_test_split.txt')
    Birds.summary()
    dataset = Birds.get_split('test', batch_size=25)

    model = MaskGenerator(n_classes=2)
    n_epochs = 1
    model.load_weights('Models/epoch_' + str(n_epochs))

    # iterate over batches
    for batch_id, (batch_images, batch_labels) in enumerate(dataset):
        batch_predictions = model(batch_images)
        for image, label in zip(batch_predictions, batch_labels):
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(tf.argmax(image, 2).numpy(), cmap='gray')
            ax[1].imshow(label.numpy()[:, :, 0], cmap='gray')
            plt.show()

