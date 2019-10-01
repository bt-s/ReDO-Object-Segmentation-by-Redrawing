import tensorflow as tf
from segmentation_network import MaskGenerator
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
    training_dataset = dataset.get_split(split='training', batch_size=10, shuffle=True)
    validation_dataset = dataset.get_split(split='validation', batch_size=10)

    # create dataset dict for train function
    datasets = {'train': training_dataset, 'val': validation_dataset}

    # number of classes in dataset | required for number of generator networks
    n_classes = dataset.n_classes

    # initialization gain for orthogonal initialization
    init_gain = 0.8

    ##########################
    # create network objects #
    ##########################

    # segmentation network
    segmentation_network = MaskGenerator(n_classes=dataset.n_classes, init_gain=init_gain)

    # dictionary of generator networks for each class
    generator_networks = {str(k): Generator(init_gain=init_gain, input_dim=32, base_channels=32) for k in
                          range(n_classes)}

    # discriminator network
    discriminator_network = Discriminator(init_gain=init_gain)

    # information conservation network
    information_network = InformationConservationNetwork(init_gain=init_gain, n_classes=n_classes,
                                                         output_dim=32)

    # dictionary of all relevant networks for adversarial training
    models = {'F': segmentation_network, 'G': generator_networks, 'D': discriminator_network,
              'Delta': information_network}

    # set names of networks
    models['F'].set_name('Segmentation_Network')
    models['D'].set_name('Discriminator')
    models['Delta'].set_name('Information Network')
    for k, G_k in generator_networks.items():
        G_k.set_name('Generator_' + str(k))
        G_k.set_region(int(k))

    # define loss function
    loss = UnsupervisedLoss(lambda_z=5)

    # define optimizer
    generator_optimizer = Adam(learning_rate=1e-4, beta_1=0, beta_2=0.9)
    discriminator_optimizer = Adam(learning_rate=1e-4, beta_1=0, beta_2=0.9)
    mask_network_optimizer = Adam(learning_rate=1e-5, beta_1=0, beta_2=0.9)
    information_network_optimizer = Adam(learning_rate=1e-4, beta_1=0, beta_2=0.9)

    # set training parameters
    n_epochs = 100
    early_stopping = EarlyStopping(patience=20, verbose=True, improvement='down')

    # define metrics dictionary
    metrics = {'train_loss_gen': Mean(), 'train_loss_gen_dis': Mean(), 'train_loss_gen_inf': Mean(),
               'train_loss_dis': Mean(), 'train_loss_dis_real': Mean(),
               'train_loss_dis_fake': Mean(), 'val_loss_gen': Mean(), 'val_loss_gen_dis': Mean(),
               'val_loss_gen_inf': Mean(),
               'val_loss_dis': Mean(), 'val_loss_dis_real': Mean(), 'val_loss_dis_fake': Mean()}

    # save tensorboard logs
    train_log_dir = 'Tensorboard_Logs/' + session_name + '/training'
    validation_log_dir = 'Tensorboard_Logs/' + session_name + '/validation'
    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(validation_log_dir)
    tensorboard_writers = {'train_writer': train_writer, 'val_writer': val_writer}

    ########################
    # Discriminator Update #
    ########################

    @tf.function
    def discriminator_update(batch_images, phase):

        # activate gradient tape
        with tf.GradientTape() as tape:

            # get segmentation masks
            batch_mask_predictions = models['F'](batch_images)

            # number of regions
            n_regions = batch_mask_predictions.shape[3]

            # sample region uniformly
            region_id = str(np.random.randint(0, n_regions))

            # redraw sampled region of image
            batch_images_redrawn, z_k = models['G'][region_id](batch_images, batch_mask_predictions, training=True)

            # get discriminator scores for real and fake image
            discriminator_score_fake = models['D'](batch_images_redrawn)
            discriminator_score_real = models['D'](batch_images)

            # compute discriminator loss for current batch
            discriminator_loss_real, discriminator_loss_fake = \
                loss.get_discriminator_loss(discriminator_score_fake, discriminator_score_real)

            discriminator_loss = discriminator_loss_fake + discriminator_loss_real

        if phase == 'train':
            # compute gradients
            discriminator_gradients = tape.gradient(discriminator_loss, models['D'].trainable_variables)
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, models['D'].trainable_variables))

        # update respective metric with computed loss and performance metrics
        metrics[phase + '_loss_dis_real'](discriminator_loss_real)
        metrics[phase + '_loss_dis_fake'](discriminator_loss_fake)
        metrics[phase + '_loss_dis'](discriminator_loss)


    ####################
    # Generator Update #
    ####################

    def generator_update(batch_images, phase):

        # activate gradient tape
        with tf.GradientTape() as tape:

            # get segmentation masks
            batch_mask_predictions = models['F'](batch_images)

            # number of regions
            n_regions = batch_mask_predictions.shape[3]

            # sample region uniformly
            region_id = str(np.random.randint(0, n_regions))

            # redraw sampled region of image
            composed_image, z_k = models['G'][region_id](batch_images, batch_mask_predictions, training=True)

            # get noise vector estimate
            z_k_hat = models['Delta'](composed_image, region_id)

            # get discriminator output for generated image
            discriminator_score_fake = models['D'](composed_image)

            # compute generator loss for current batch
            generator_loss_dis, generator_loss_inf = loss.get_generator_loss(discriminator_score_fake, z_k, z_k_hat)

            generator_loss = generator_loss_dis + generator_loss_inf

        if phase == 'train':
            # compute gradients
            gradients = tape.gradient(generator_loss, models['F'].trainable_variables + models['G'][
                region_id].trainable_variables + models['Delta'].trainable_variables)
            mask_network_gradients = gradients[:len(models['F'].trainable_variables)]
            generator_gradients = gradients[len(models['F'].trainable_variables):-len(models['Delta'].trainable_variables)]
            information_network_gradients = gradients[-len(models['Delta'].trainable_variables):]
     
            # update weights
            generator_optimizer.apply_gradients(zip(generator_gradients, models['G'][region_id].trainable_variables))
            mask_network_optimizer.apply_gradients(zip(mask_network_gradients, models['F'].trainable_variables))
            information_network_optimizer.apply_gradients(zip(information_network_gradients, models['Delta'].trainable_variables))

        # update respective metric with computed loss and performance metrics
        metrics[phase + '_loss_gen_dis'](generator_loss_dis)
        metrics[phase + '_loss_gen_inf'](generator_loss_inf)
        metrics[phase + '_loss_gen'](generator_loss)

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

            # print progress
            print('Phase: {:s}'.format(phase))

            # iterate over batches
            for batch_id, (batch_images, batch_labels) in enumerate(datasets[phase]):

                # print progress
                print('Batch: {:d}/{:d}'.format(batch_id + 1, tf.data.experimental.cardinality(datasets[phase])))

                if batch_id % 2:
                    print('Discriminator Update!')
                    discriminator_update(batch_images, phase)
                else:
                    print('Generator Update!')
                    generator_update(batch_images, phase)

        # log epoch, print summary, evaluate early stopping
        train_utils.log_epoch(metrics, tensorboard_writers, epoch, scheme='unsupervised')

        # call early stopping module
        early_stopping(metrics['val_loss_gen'].result()+metrics['val_loss_dis'].result(), epoch, session_name, models)

        # reset the metrics for the next epoch
        [metric.reset_states() for metric in metrics.values()]

        if epoch == 0:
            for model in models.values():
                if isinstance(model, collections.Mapping):
                    for sub_model in model.values():
                        print(sub_model.model_name + ', number of variables: ', len(sub_model.trainable_variables))
                else:
                    print(model.model_name + ', number of variables: ', len(model.trainable_variables))
