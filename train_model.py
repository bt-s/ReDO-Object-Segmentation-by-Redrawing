import tensorflow as tf
from networks import MaskGenerator
from datasets import BirdDataset, FlowerDataset, FaceDataset
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # create datasets
    dataset = BirdDataset(root='Datasets/Birds/', image_dir='images/', label_dir='labels/', path_file='paths.txt',
                        split_file='train_val_test_split.txt')
    training_dataset = dataset.get_split(split='training', batch_size=25)
    validation_dataset = dataset.get_split(split='validation', batch_size=25)

    # create dataset dict for train function
    datasets = {'train': training_dataset, 'val': validation_dataset}

    # create network object
    model = MaskGenerator(n_classes=2)

    # define loss function
    loss = SparseCategoricalCrossentropy()

    # define optimizer
    optimizer = Adam(learning_rate=1e-5, beta_1=0, beta_2=0.9)

    # set training parameters
    n_epochs = 1

    # define metrics dictionary
    metrics = {'train_loss': Mean(), 'val_loss': Mean()}

    # compute one training step
    @tf.function
    def step(batch_images, batch_labels, phase):

        if phase == 'train':
            # activate gradient tape
            with tf.GradientTape() as tape:
                # get predictions for current batch
                batch_predictions = model(batch_images)
                # compute loss for current batch
                batch_loss = loss(batch_labels, batch_predictions)
                # compute gradients
                gradients = tape.gradient(batch_loss, model.trainable_variables)
                # update weights
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        else:
            # get predictions for current batch
            batch_predictions = model(batch_images)
            # compute loss for current batch
            batch_loss = loss(batch_labels, batch_predictions)

        return batch_loss


    # iterate over specified number of epochs
    for epoch in range(n_epochs):

        # each epoch consists of two phases: training and validation
        phases = ['train', 'val']
        for phase in phases:

            # iterate over batches
            for batch_id, (batch_images, batch_labels) in enumerate(datasets[phase]):

                # take one step
                batch_loss = step(batch_images, batch_labels, phase=phase)

                # update respective metric with computed loss
                metrics[phase + '_loss'](batch_loss)

        # print summary at the end of epoch
        epoch_summary = 'Epoch {}, Train Loss: {}, Val Loss: {}'
        print(epoch_summary.format(epoch + 1, metrics['train_loss'].result(), metrics['val_loss'].result()))

        # reset the metrics for the next epoch
        metrics['train_loss'].reset_states()
        metrics['val_loss'].reset_states()

        model.save_weights('Weights/' + str(dataset.type) + '_epoch_' + str(epoch))
