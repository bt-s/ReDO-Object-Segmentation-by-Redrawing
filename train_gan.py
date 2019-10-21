#!/usr/bin/python3

"""train_gan.py - Given dataset and other hyper-parameters, train all networks:
    - generator(s)
    - discriminator
    - information
    - mask/segmentation

For the NeurIPS Reproducibility Challange and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chiemelewski-Anders, Mats Steinweg & Bas Straathof"


import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from argparse import ArgumentParser, Namespace
from sys import argv
import logging
from typing import Dict, Tuple

from datasets import BirdDataset, FlowerDataset, FaceDataset
from train_utils import UnsupervisedLoss, log_epoch
from generator import Generator
from discriminator import Discriminator
from segmentation_network import SegmentationNetwork
from information_network import InformationConservationNetwork

SUPPORTED_DATASETS = {'flowers': FlowerDataset, 'birds': BirdDataset}


def parse_train_args():
    """Parses CL arguments"""
    parser = ArgumentParser()
    # Positional arguments
    parser.add_argument('dataset', choices=SUPPORTED_DATASETS.keys())

    # Options/flags
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-g', '--init-gain', type=float, default=0.8)
    parser.add_argument('-w', '--weight-decay', type=float, default=1e-4)
    parser.add_argument('-lz', '--lambda-z', type=float, default=5.0,
                        help=('Multiplicative factor for information'
                              'conservation loss'))
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-c', '--checkpoint-iter', type=int, default=350)
    parser.add_argument('-s', '--session-name', type=str, default='MySession')
    parser.add_argument('-z', '--z-dim', type=int, default=32,
            help='Dimension of latent z-variable')
    parser.add_argument('-ch', '--base-channels', type=int, default=32,
            help='Dataset-dependent constant for number of channels in network')
    parser.add_argument('-b1', '--beta-1', type=float, default=0.0)
    parser.add_argument('-b2', '--beta-2', type=float, default=0.9)
    parser.add_argument('-lr1', '--learning-rate-other', type=float,
            default=1e-4)
    parser.add_argument('-lr2', '--learning-rate-mask', type=float,
            default=1e-5)
    parser.add_argument('-l', '--log-level', type=int, default=30)

    return parser.parse_args(argv[1:])


def discriminator_update(batch_images_real: tf.Tensor, training: bool, optimizers: Dict,
        models: Dict, metrics: Dict, adversarial_loss: UnsupervisedLoss,
        phase: str):
    """Updates the real and fake disciminator losses

    Args:
        batch_images_real: Image batch (of size n) of shape (n, 128, 128, 3)
        training: Whether we are training
        models: Dict of models
        metrics: Dict of metrics
        adversarial_loss: Loss
    """
    with tf.GradientTape() as tape:
        # Get segmentation masks
        batch_masks_logits = models['F'](batch_images_real)

        # Get fake images from generator
        # The number of images generated = batch_size * n_classes
        batch_images_fake = models['G'](batch_images_real,
                batch_masks_logits, update_generator=False,
                training=training)

        # Get logits for real and fake images
        d_logits_real = models['D'](batch_images_real, training)
        d_logits_fake = models['D'](batch_images_fake, training)

        # Compute discriminator loss for current batch
        d_loss_real, d_loss_fake = adversarial_loss.get_d_loss(
            d_logits_real, d_logits_fake)

        d_loss = d_loss_real + d_loss_fake
        print('Discriminator loss (real): ', d_loss_real)
        print('Discriminator loss (fake): ', d_loss_fake)

    if training:
        # Compute gradients
        d_gradients = tape.gradient(d_loss, models['D'].trainable_variables)
        optimizers['D'].apply_gradients(zip(d_gradients,
            models['D'].trainable_variables))

    # Update summary with the computed loss
    metrics['d_r_loss_' + phase](d_loss_real)
    metrics['d_f_loss_' + phase](d_loss_fake)


def generator_update(batch_images_real: tf.Tensor, training: bool,
        models: Dict, metrics: Dict, optimizers: Dict,
        adversarial_loss: UnsupervisedLoss, phase: str):
    """Updates the generator and information losses

    Args:
        batch_images_real: Image batch (of size n) of shape (n, 128, 128, 3)
        training: Whether we are training
        models: Dict of models
        metrics: Dict of metrics
        optimizers: Dict of optimizers
        adversarial_loss: Loss
    """
    with tf.GradientTape(persistent=True) as tape:
        # Get segmentation masks
        batch_masks = models['F'](batch_images_real)

        # Get fake images from generator
        # Number of images generated = batch_size * n_classes
        batch_images_fake, batch_regions_fake, batch_z_k, k = models['G'](
                batch_images_real, batch_masks, update_generator=True,
                training=training)
        # Get the recovered z-value from the information network
        batch_z_k_hat = models['I'](batch_regions_fake, training=training)

        # Get logits for fake images
        d_logits_fake = models['D'](batch_images_fake, training)

        # Compute generator loss for current batch
        g_loss_d, g_loss_i = adversarial_loss.get_g_loss(d_logits_fake,
                batch_z_k, batch_z_k_hat)

        g_loss = g_loss_d + g_loss_i
        print('Generator loss (discriminator): ', g_loss_d)
        print('Discriminator loss (information): ', g_loss_i)

    if training:
        # Compute gradients
        gradients = tape.gradient(g_loss, models['F'].trainable_variables +
                models['G'].class_generators[k].trainable_variables)

        f_gradients = gradients[:len(models['F'].trainable_variables)]
        g_gradients = gradients[-len(models['G'].class_generators[k].trainable_variables):]

        i_gradients = tape.gradient(g_loss_i,
                models['I'].trainable_variables)

        # Update weights
        optimizers['G'].apply_gradients(zip(g_gradients,
            models['G'].class_generators[k].trainable_variables))
        optimizers['F'].apply_gradients(zip(f_gradients,
            models['F'].trainable_variables))
        optimizers['I'].apply_gradients(zip(i_gradients,
            models['I'].trainable_variables))

    # Update summary with computed loss
    metrics['g_d_loss_' + phase](g_loss_d)
    metrics['g_i_loss_' + phase](g_loss_i)


def create_network_objects(args: Namespace) -> Dict:
    """Create, and initialize, all necessary networks for training

    Args:
        args: Dictionary of CL arguments

    Returns:
        models: Segmentation, generator, discriminator and information networks
    """
    segmentation_network = SegmentationNetwork(n_classes=args.n_classes,
            init_gain=args.init_gain, weight_decay=args.weight_decay)

    generator = Generator(n_classes=args.n_classes, n_input=args.z_dim,
            init_gain=args.init_gain, base_channels=args.base_channels)

    discriminator = Discriminator(init_gain=args.init_gain)

    information_network = InformationConservationNetwork(
            init_gain=args.init_gain, n_classes=args.n_classes,
            n_output=args.z_dim)

    models = {'F': segmentation_network, 'G': generator, 'D': discriminator,
            'I': information_network}

    return models


def train(args: Namespace, datasets: Dict):
    """Trains the generator, discriminator, maks and information network

    Args:
        args: CL input arguments
        datasets: Training and validation data sets
    """
    # Initialize the network objects
    models = create_network_objects(args)

    # Initialize the adversarial loss
    adversarial_loss = UnsupervisedLoss(lambda_z=args.lambda_z)

    # Define optimizers
    g_optimizer = Adam(learning_rate=args.learning_rate_other,
            beta_1=args.beta_1, beta_2=args.beta_2)
    d_optimizer = Adam(learning_rate=args.learning_rate_other,
            beta_1=args.beta_1, beta_2=args.beta_2)
    i_optimizer = Adam(learning_rate=args.learning_rate_other,
            beta_1=args.beta_1, beta_2=args.beta_2)
    f_optimizer = Adam(learning_rate=args.learning_rate_mask,
            beta_1=args.beta_1, beta_2=args.beta_2)

    optimizers = {'G': g_optimizer, 'D': d_optimizer, 'I': i_optimizer,
            'F': f_optimizer}

    # Define metrics dictionary
    metrics = {'g_d_loss_train': Mean(), 'g_i_loss_train': Mean(),
            'd_r_loss_train': Mean(), 'd_f_loss_train': Mean(),
            'g_d_loss_val': Mean(), 'g_i_loss_val': Mean(),
            'd_r_loss_val': Mean(), 'd_f_loss_val': Mean()}

    # Save tensorboard logs
    train_log_dir = 'Tensorboard_Logs/' + args.session_name + '/training'
    validation_log_dir = 'Tensorboard_Logs/' + args.session_name + '/validation'

    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(validation_log_dir)

    tensorboard_writers = {'train_writer': train_writer,
            'val_writer': val_writer}

    # Iteratively train the networks
    for epoch in range(args.epochs):
        # Print progress
        print('###########################################################')
        #print(f'Epoch: {epoch + 1}')
        print('Epoch: ', epoch+1)

        # Each epoch consists of two phases: training and validation
        phases = ['train', 'val']
        for phase in phases:
            training = True if phase == 'train' else False

            # Print progress
            #print(f'Phase: {phase}')
            print('Phase: ', phase)

            # Iterate over batches
            for batch_id, (batch_images_real, _) in enumerate(datasets[phase]):

                # Print progress
                ds_len = tf.data.experimental.cardinality(datasets[phase])
                #print(f'Batch: {batch_id + 1} / {ds_len}')
                print('Batch {:d}/{:d}'.format(batch_id+1, ds_len))

                if (batch_id % 2) == 0:
                    # Update generator
                    generator_update(batch_images_real, training, models,
                            metrics, optimizers, adversarial_loss, phase=phase)
                else:
                    # Update discriminator
                    discriminator_update(batch_images_real, training, optimizers,
                            models, metrics, adversarial_loss,
                            phase=phase)

                # Save model weights
                if (batch_id + 1) % args.checkpoint_iter == 0:
                    #for model in models.values():
                        #model.save_weights(f'Weights/{args.session_name}/' \
                         #       f'{model.model_name}/Epoch_{str(epoch+1)}' \
                          #      f'batch_{str(batch_id+1)}')
                        models['F'].save_weights(
                            'Weights/' + args.session_name + '/' +
                            models['F'].model_name + '/Epoch_' + str(epoch + 1) +
                            '_Batch_' + str(batch_id + 1) + '/'
                        )

        # Log epoch for tensorboard and print summary
        log_epoch(metrics, tensorboard_writers, epoch, scheme='unsupervised')

        # Reset all metrics for the next epoch
        [metric.reset_states() for metric in metrics.values()]


def main(args: Namespace):
    tf.get_logger().setLevel(args.log_level)

    # Get datasets
    dataset = SUPPORTED_DATASETS[args.dataset]()

    # Split dataset into training and validation sets
    # Note: there is no test set, since this is an unsupervised learning approach
    training_dataset = dataset.get_split(split='training',
            batch_size=args.batch_size, shuffle=True)

    validation_dataset = dataset.get_split(split='validation',
            batch_size=args.batch_size)

    # Create dataset dict for train function
    datasets = {'train': training_dataset, 'val': validation_dataset}

    # Number of classes in dataset | required for number of class generators
    args.n_classes = dataset.n_classes

    # Train the generator, discriminator, mask and information networks
    train(args, datasets)


if __name__ == '__main__':
    main(parse_train_args())

