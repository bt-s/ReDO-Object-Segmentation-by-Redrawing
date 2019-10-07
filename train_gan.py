# import tensorflow utilities
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
# import datasets
from datasets import BirdDataset, FlowerDataset, FaceDataset
# import training modules and functions
from train_utils import UnsupervisedLoss, log_epoch
# import networks
from generator import Generator
from discriminator import Discriminator
from segmentation_network import SegmentationNetwork


if __name__ == '__main__':

    # session name | name of directories for tensorboard logs and saved models
    session_name = 'Unsupervised_Flowers'

    # set batch size
    batch_size = 8

    # create datasets
    dataset = FlowerDataset()
    training_dataset = dataset.get_split(split='training', batch_size=batch_size, shuffle=True)
    validation_dataset = dataset.get_split(split='validation', batch_size=batch_size)

    # create dataset dict for train function
    datasets = {'train': training_dataset, 'val': validation_dataset}

    # number of classes in dataset | required for number of class generators
    n_classes = dataset.n_classes

    # initialization gain for orthogonal initialization
    init_gain = 0.8

    # weight decay factor for segmentation network
    weight_decay = 1e-4

    # dimensionality of the sampled noise vector used as input to class generators
    n_input = 32

    # dataset-dependent constant for number of channels in network
    base_channels = 32

    ##########################
    # create network objects #
    ##########################

    # segmentation network
    segmentation_network = SegmentationNetwork(n_classes=dataset.n_classes, init_gain=init_gain,
                                               weight_decay=weight_decay)

    # generator object containing class generator objects for each class and information conservation network
    generator = Generator(n_classes=n_classes, n_input=n_input, init_gain=init_gain, base_channels=base_channels)

    # discriminator network
    discriminator = Discriminator(init_gain=init_gain)

    # dictionary of all relevant networks for adversarial training
    models = {'F': segmentation_network, 'G': generator, 'D': discriminator}

    # loss function
    lambda_z = 5.0  # multiplicative factor for information conservation loss
    adversarial_loss = UnsupervisedLoss(lambda_z=lambda_z)

    # define optimizer
    g_optimizer = Adam(learning_rate=1e-4, beta_1=0, beta_2=0.9)  # optimizer for the generator
    d_optimizer = Adam(learning_rate=1e-4, beta_1=0, beta_2=0.9)  # optimizer for the discriminator
    f_optimizer = Adam(learning_rate=1e-5, beta_1=0, beta_2=0.9)  # optimizer for the segmentation network

    # set number of training epochs
    n_epochs = 100

    # define metrics dictionary
    metrics = {'g_d_loss_train': Mean(), 'g_i_loss_train': Mean(),  # generator loss: discriminator | information
               'd_r_loss_train': Mean(), 'd_f_loss_train': Mean(),  # discriminator loss: real | fake
               'g_d_loss_val': Mean(), 'g_i_loss_val': Mean(),
               'd_r_loss_val': Mean(), 'd_f_loss_val': Mean()}

    # save tensorboard logs
    train_log_dir = 'Tensorboard_Logs/' + session_name + '/training'
    validation_log_dir = 'Tensorboard_Logs/' + session_name + '/validation'
    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(validation_log_dir)
    tensorboard_writers = {'train_writer': train_writer, 'val_writer': val_writer}

    # initialize weights
    batch_images_init = tf.random.normal([batch_size, 128, 128, 3])
    batch_masks_init = tf.random.normal([batch_size, 128, 128, 2])
    for k in range(n_classes):
        batch_images_fake, z_k, z_k_hat = models['G'](batch_images_init, batch_masks_init, k=k, update_generator=True,
                                      training=True)

    ########################
    # Discriminator Update #
    ########################

    def discriminator_update(batch_images_real, k, training):

        # activate gradient tape
        with tf.GradientTape() as tape:

            # get segmentation masks
            batch_masks_logits = models['F'](batch_images_real)

            # get fake images from generator | number of images generated = batch_size * n_classes
            batch_images_fake = models['G'](batch_images_real, batch_masks_logits, k=k, update_generator=False, training=training)

            # get logits for real and fake images
            d_logits_real = models['D'](batch_images_real, training)
            d_logits_fake = models['D'](batch_images_fake, training)

            # compute discriminator loss for current batch
            d_loss_real, d_loss_fake = adversarial_loss.get_d_loss(d_logits_real, d_logits_fake)
            d_loss = d_loss_real + d_loss_fake
            print('D_R: ', d_loss_real)
            print('D_F: ', d_loss_fake)

        if training:
            # compute gradients
            d_gradients = tape.gradient(d_loss, models['D'].trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradients, models['D'].trainable_variables))

        # update summary with computed loss
        metrics['d_r_loss_' + phase](d_loss_real)
        metrics['d_f_loss_' + phase](d_loss_fake)

    ####################
    # Generator Update #
    ####################

    def generator_update(batch_images_real, k, training):

        # activate gradient tape
        with tf.GradientTape() as tape:

            # get segmentation masks
            batch_masks_logits = models['F'](batch_images_real)

            # get fake images from generator | number of images generated = batch_size * n_classes
            batch_images_fake, batch_z_k, batch_z_k_hat = \
                models['G'](batch_images_real, batch_masks_logits, k, update_generator=True, training=training)

            # get logits for fake images
            d_logits_fake = models['D'](batch_images_fake, training)

            # compute generator loss for current batch
            g_loss_d, g_loss_i = adversarial_loss.get_g_loss(d_logits_fake, batch_z_k, batch_z_k_hat)
            g_loss = g_loss_d + g_loss_i
            print('G_D: ', g_loss_d)
            print('G_I: ', g_loss_i)

        if training:
            # compute gradients
            gradients = tape.gradient(g_loss, models['F'].trainable_variables + models['G'].trainable_variables)
            f_gradients = gradients[:len(models['F'].trainable_variables)]
            g_gradients = gradients[-len(models['G'].trainable_variables):]

            # update weights
            g_optimizer.apply_gradients(zip(g_gradients, models['G'].trainable_variables))
            f_optimizer.apply_gradients(zip(f_gradients, models['F'].trainable_variables))

        # update summary with computed loss
        metrics['g_d_loss_' + phase](g_loss_d)
        metrics['g_i_loss_' + phase](g_loss_i)

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
            for batch_id, (batch_images_real, _) in enumerate(datasets[phase]):

                # print progress
                print('Batch: {:d}/{:d}'.format(batch_id + 1, tf.data.experimental.cardinality(datasets[phase])))

                for k in range(n_classes):

                    print('Region: ', k)

                    # update generator
                    generator_update(batch_images_real, k, training)

                    # update discriminator
                    discriminator_update(batch_images_real, k, training)

                # save model weights
                if ((epoch+1)*batch_id) % 200 == 0:
                    for model in models.values():
                        model.save_weights(
                            'Weights/' + session_name + '/' + model.model_name + '/Iteration_' + str((epoch+1)*batch_id) + '/')

        # log epoch for tensorboard and print summary
        log_epoch(metrics, tensorboard_writers, epoch, scheme='unsupervised')

        # reset all metrics for the next epoch
        [metric.reset_states() for metric in metrics.values()]

