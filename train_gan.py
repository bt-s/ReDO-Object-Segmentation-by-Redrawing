import tensorflow as tf
from segmentation_network import SegmentationNetwork
from datasets import BirdDataset, FlowerDataset, FaceDataset
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean, Accuracy, MeanIoU
import train_utils
from tensorflow.keras.layers import Softmax, Dense
from train_utils import UnsupervisedLoss, EarlyStopping
from generator import Generator
from discriminator import Discriminator
from information_network import InformationConservationNetwork
import numpy as np
import collections

if __name__ == '__main__':

    # session name
    session_name = 'Unsupervised_Flowers'

    # create datasets
    dataset = FlowerDataset()
    training_dataset = dataset.get_split(split='training', batch_size=8, shuffle=True)
    validation_dataset = dataset.get_split(split='validation', batch_size=8)

    # create dataset dict for train function
    datasets = {'train': training_dataset, 'val': validation_dataset}

    # number of classes in dataset | required for number of generator networks
    n_classes = dataset.n_classes

    # initialization gain for orthogonal initialization
    init_gain = 0.9

    # weight decay factor
    weight_decay = 1e-4

    ##########################
    # create network objects #
    ##########################

    # segmentation network
    segmentation_network = SegmentationNetwork(n_classes=dataset.n_classes, init_gain=init_gain, weight_decay=weight_decay)

    # dictionary of generator networks for each class
    generator_networks = {str(k): Generator(init_gain=init_gain, base_channels=32) for k in
                          range(n_classes)}

    # discriminator network
    discriminator_network = Discriminator(init_gain=init_gain)

    # information conservation network
    information_network = InformationConservationNetwork(init_gain=init_gain, n_classes=n_classes,
                                                         output_dim=32)

    # dictionary of all relevant networks for adversarial training
    models = {'F': segmentation_network, 'G': generator_networks, 'D': discriminator_network,
              'I': information_network}

    # set names of networks
    models['F'].set_name('Segmentation_Network')
    models['D'].set_name('Discriminator')
    models['I'].set_name('Information Network')
    for k, G_k in generator_networks.items():
        G_k.set_name('Generator_' + str(k))
        G_k.set_region(int(k))

    # define loss function
    loss = UnsupervisedLoss(lambda_z=5)

    # define optimizer
    g_optimizer = Adam(learning_rate=1e-4, beta_1=0, beta_2=0.9)
    d_optimizer = Adam(learning_rate=1e-4, beta_1=0, beta_2=0.9)
    f_optimizer = Adam(learning_rate=1e-5, beta_1=0, beta_2=0.9)
    i_optimizer = Adam(learning_rate=1e-4, beta_1=0, beta_2=0.9)

    # set training parameters
    n_epochs = 100
    early_stopping = EarlyStopping(patience=20, verbose=True, improvement='down')

    # define metrics dictionary
    metrics = {'g_loss_train': Mean(), 'g_d_loss_train': Mean(), 'g_i_loss_train': Mean(),
               'd_loss_train': Mean(), 'd_r_loss_train': Mean(), 'd_f_loss_train': Mean(),
               'g_loss_val': Mean(), 'g_d_loss_val': Mean(), 'g_i_loss_val': Mean(),
               'd_loss_val': Mean(), 'd_r_loss_val': Mean(), 'd_f_loss_val': Mean()}

    # save tensorboard logs
    train_log_dir = 'Tensorboard_Logs/' + session_name + '/training'
    validation_log_dir = 'Tensorboard_Logs/' + session_name + '/validation'
    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(validation_log_dir)
    tensorboard_writers = {'train_writer': train_writer, 'val_writer': val_writer}

    ########################
    # Discriminator Update #
    ########################

    def discriminator_update(batch_images_real, training):

        # activate gradient tape
        with tf.GradientTape() as tape:

            # get segmentation masks
            batch_mask_predictions = models['F'](batch_images_real)

            # number of regions
            n_regions = batch_mask_predictions.shape[3]

            batch_images_fake = None
            # redraw sampled region of image
            for k in range(n_regions):
                batch_images_k_fake, _ = models['G'][str(k)](batch_images_real, batch_mask_predictions, 32, training=training)

                if batch_images_fake is None:
                    batch_images_fake = batch_images_k_fake
                else:
                    batch_images_fake = tf.concat((batch_images_fake, batch_images_k_fake), axis=0)

            # get discriminator scores for real and fake image
            d_logits_real = models['D'](batch_images_real, training)
            d_logits_fake = models['D'](batch_images_fake, training)

            # compute discriminator loss for current batch
            d_loss_real, d_loss_fake = loss.get_discriminator_loss(d_logits_real, d_logits_fake)
            d_loss = d_loss_real + d_loss_fake
            print('D_R: ', d_loss_real)
            print('D_F: ', d_loss_fake)

        if training:
            # compute gradients
            d_gradients = tape.gradient(d_loss, models['D'].trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradients, models['D'].trainable_variables))

            # update respective metric with computed loss and performance metrics
        metrics['d_r_loss_' + phase](d_loss_real)
        metrics['d_f_loss_' + phase](d_loss_fake)
        metrics['d_loss_' + phase](d_loss)

    ####################
    # Generator Update #
    ####################

    def generator_update(batch_images_real, training):

        # activate gradient tape
        with tf.GradientTape() as tape:

            # get segmentation masks
            batch_mask_predictions = models['F'](batch_images_real)

            # number of regions
            n_regions = batch_mask_predictions.shape[3]

            batch_images_fake = None
            batch_z_k_hat = None
            batch_z_k = None
            # redraw sampled region of image
            for k in range(n_regions):
                batch_images_k_fake, z_k = models['G'][str(k)](batch_images_real, batch_mask_predictions, 32,
                                                             training=training)

                # get noise vector estimate
                z_k_hat = models['I'](batch_images_k_fake, str(k), training)

                if batch_images_fake is None:
                    batch_images_fake = batch_images_k_fake
                    batch_z_k_hat = z_k_hat
                    batch_z_k = z_k
                else:
                    batch_images_fake = tf.concat((batch_images_fake, batch_images_k_fake), axis=0)
                    batch_z_k_hat = tf.concat((batch_z_k_hat, z_k_hat), axis=0)
                    batch_z_k = tf.concat((batch_z_k, z_k), axis=0)

            # get discriminator output for generated image
            d_logits_fake = models['D'](batch_images_fake, training)

            # compute generator loss for current batch
            g_loss_d, g_loss_i = loss.get_generator_loss(d_logits_fake, batch_z_k, batch_z_k_hat)
            g_loss = g_loss_d + g_loss_i
            print('G_D: ', g_loss_d)
            print('G_I: ', g_loss_i)

        if training:
            # compute gradients
            gradients = tape.gradient(g_loss, models['F'].trainable_variables + models['G'][
                '0'].trainable_variables + models['G'][
                '1'].trainable_variables + models['I'].trainable_variables)
            f_gradients = gradients[:len(models['F'].trainable_variables)]
            g_gradients = gradients[len(models['F'].trainable_variables):-len(models['I'].trainable_variables)]
            i_gradients = gradients[-len(models['I'].trainable_variables):]

            # update weights
            g_optimizer.apply_gradients(zip(g_gradients, models['G']['0'].trainable_variables +
                                         models['G']['1'].trainable_variables))
            f_optimizer.apply_gradients(zip(f_gradients, models['F'].trainable_variables))
            i_optimizer.apply_gradients(zip(i_gradients, models['I'].trainable_variables))

            # update respective metric with computed loss and performance metrics
        metrics['g_d_loss_' + phase](g_loss_d)
        metrics['g_i_loss_' + phase](g_loss_i)
        metrics['g_loss_' + phase](g_loss)

    #################
    # Training Loop #
    #################

    for epoch in range(n_epochs):

        # print progress
        print('###########################################################')
        print('Epoch: {:d}'.format(epoch + 1))

        # each epoch consists of two phases: training and validation
        phases = ['train', 'val']
        for phase in phases:

            if phase == 'train':
                training = True
            else:
                training = False

            # print progress
            print('Phase: {:s}'.format(phase))

            # iterate over batches
            for batch_id, (batch_images_real, batch_labels) in enumerate(datasets[phase]):

                # print progress
                print('Batch: {:d}/{:d}'.format(batch_id + 1, tf.data.experimental.cardinality(datasets[phase])))

                # update discriminator
                discriminator_update(batch_images_real, training)

                # update generator
                generator_update(batch_images_real, training)

        # log epoch, print summary, evaluate early stopping
        train_utils.log_epoch(metrics, tensorboard_writers, epoch, scheme='unsupervised')

        # call early stopping module
        early_stopping(metrics['g_loss_val'].result() + metrics['d_loss_val'].result(), epoch, session_name, models)

        # reset the metrics for the next epoch
        [metric.reset_states() for metric in metrics.values()]

        if epoch == 0:
            for model in models.values():
                if isinstance(model, collections.Mapping):
                    for sub_model in model.values():
                        print(sub_model.model_name + ', number of variables: ', len(sub_model.trainable_variables))
                else:
                    print(model.model_name + ', number of variables: ', len(model.trainable_variables))
