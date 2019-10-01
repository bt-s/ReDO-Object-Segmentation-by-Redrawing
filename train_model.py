import tensorflow as tf
from segmentation_network import MaskGenerator
from datasets import BirdDataset, FlowerDataset, FaceDataset
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean, Accuracy, MeanIoU
import train_utils
from time import time
from train_utils import SupervisedLoss, EarlyStopping

if __name__ == '__main__':

    # create datasets
    dataset = FaceDataset()
    training_dataset = dataset.get_split(split='validation', size=400, batch_size=25, shuffle=True)
    validation_dataset = dataset.get_split(split='test', batch_size=25)

    # create dataset dict for train function
    datasets = {'train': training_dataset, 'val': validation_dataset}

    # initializer
    initializer = tf.initializers.orthogonal(gain=0.8)

    # create network object
    model = MaskGenerator(n_classes=dataset.n_classes, initializer=initializer)
    model.set_save_name('Supervised_Faces_400')

    # define loss function
    loss = SupervisedLoss()

    # define optimizer
    optimizer = Adam(learning_rate=1e-5, beta_1=0, beta_2=0.9)

    # set training parameters
    n_epochs = 100
    early_stopping = EarlyStopping(patience=7, verbose=True)

    # define metrics dictionary
    metrics = {'train_loss': Mean(), 'train_accuracy': Mean(), 'train_IoU': Mean(), 'train_step_time': Mean(),
               'val_loss': Mean(), 'val_accuracy': Mean(), 'val_IoU': Mean(), 'val_step_time': Mean()}

    # save tensorboard logs
    train_log_dir = 'Tensorboard_Logs/' + model.save_name + '/training'
    validation_log_dir = 'Tensorboard_Logs/' + model.save_name + '/validation'
    graph_log_dir = 'Tensorboard_Logs/' + model.save_name + '/validation'
    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(validation_log_dir)
    graph_writer = tf.summary.create_file_writer(graph_log_dir)
    tensorboard_writers = {'train_writer': train_writer, 'val_writer': val_writer, 'graph_writer': graph_writer}

    # compute one training step
    @tf.function
    def step(batch_images, batch_labels, phase):

        step_start_time = time()

        if phase == 'train':
            # tf.summary.trace_on(graph=True, profiler=True)
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
            # with graph_writer.as_default():
                # tf.summary.trace_export(name="step_function", step=epoch, profiler_outdir=graph_log_dir)

        else:
            # get predictions for current batch
            batch_predictions = model(batch_images)
            # compute loss for current batch
            batch_loss = loss(batch_predictions, batch_labels)

        # get elapsed time
        step_end_time = time()
        step_time = step_end_time - step_start_time

        # update respective metric with computed loss and performance metrics
        metrics[phase + '_loss'](batch_loss)
        batch_accuracy = train_utils.compute_accuracy(batch_predictions, batch_labels)
        batch_iou = train_utils.compute_IoU(batch_predictions, batch_labels)
        metrics[phase + '_IoU'](batch_iou)
        metrics[phase + '_accuracy'](batch_accuracy)
        metrics[phase + '_step_time'](step_time)

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
        train_utils.log_epoch(metrics, tensorboard_writers, epoch, scheme='supervised')

        # call early stopping module
        early_stopping(metrics['val_loss'].result(), epoch, model)

        # reset the metrics for the next epoch
        [metric.reset_states() for metric in metrics.values()]

