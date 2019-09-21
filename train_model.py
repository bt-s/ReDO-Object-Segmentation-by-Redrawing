import tensorflow as tf
from networks import MaskGenerator
from datasets import BirdDataset, FlowerDataset, FaceDataset
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean, Accuracy, MeanIoU
import datetime
import train_utils
from train_utils import SupervisedLoss, EarlyStopping

if __name__ == '__main__':

    # create datasets
    dataset = BirdDataset(root='Datasets/Birds/', image_dir='images/', label_dir='labels/', path_file='paths.txt',
                        split_file='train_val_test_split.txt')
    training_dataset = dataset.get_split(split='training', batch_size=25)
    validation_dataset = dataset.get_split(split='validation', batch_size=25)

    # create dataset dict for train function
    datasets = {'train': training_dataset, 'val': validation_dataset}

    # create network object
    model = MaskGenerator(n_classes=dataset.n_classes)
    model.set_save_name('MaskGenerator_Birds')

    # define loss function
    loss = SupervisedLoss()

    # define optimizer
    optimizer = Adam(learning_rate=1e-5, beta_1=0, beta_2=0.9)

    # set training parameters
    n_epochs = 10
    early_stopping = EarlyStopping(patience=20, verbose=True)

    # define metrics dictionary
    metrics = {'train_loss': Mean(), 'train_accuracy': Mean(), 'train_IoU': Mean(),
               'val_loss': Mean(), 'val_accuracy': Mean(), 'val_IoU': Mean()}
    # batch metrics
    accuracy = Accuracy()
    iou = MeanIoU(model.n_classes)

    # save tensorboard logs
    train_log_dir = 'Tensorboard_Logs/' + model.save_name + '/training'
    validation_log_dir = 'Tensorboard_Logs/' + model.save_name + '/validation'
    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(validation_log_dir)
    tensorboard_writers = {'train_writer': train_writer, 'val_writer': val_writer}

    # compute one training step
    @tf.function
    def step(batch_images, batch_labels, phase):

        if phase == 'train':
            # activate gradient tape
            with tf.GradientTape() as tape:
                # get predictions for current batch
                batch_predictions = model(batch_images)
                # compute loss for current batch
                batch_loss = loss(batch_predictions, batch_labels)
            # compute gradients
            gradients = tape.gradient(batch_loss, model.trainable_variables)
            # update weights
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        else:
            # get predictions for current batch
            batch_predictions = model(batch_images)
            # compute loss for current batch
            batch_loss = loss(batch_predictions, batch_labels)

        # update respective metric with computed loss
        metrics[phase + '_loss'](batch_loss)
        batch_accuracy = accuracy(tf.argmax(batch_predictions, axis=3), tf.argmax(batch_labels, axis=3))
        metrics[phase + '_accuracy'](batch_accuracy)
        batch_iou = iou(tf.argmax(batch_predictions, axis=3), tf.argmax(batch_labels, axis=3))
        metrics[phase + '_IoU'](batch_iou)

    # iterate over specified number of epochs
    for epoch in range(n_epochs):

        # print progress
        print('###########################################################')
        print('Epoch: {:d}'.format(epoch+1))

        # each epoch consists of two phases: training and validation
        phases = ['train', 'val']
        for phase in phases:

            # print progress
            print('Phase: {:s}'.format(phase))

            # iterate over batches
            for batch_id, (batch_images, batch_labels) in enumerate(datasets[phase]):

                # print progress
                print('Batch: {:d}/{:d}'.format(batch_id+1, tf.data.experimental.cardinality(datasets[phase])))

                # take one step
                step(batch_images, batch_labels, phase=phase)

        # log epoch, print summary, evaluate early stopping
        train_utils.log_epoch(metrics, tensorboard_writers, epoch)

        # call early stopping module
        early_stopping(metrics['val_loss'].result(), epoch, model)

        # reset the metrics for the next epoch
        [metric.reset_states() for metric in metrics.values()]

