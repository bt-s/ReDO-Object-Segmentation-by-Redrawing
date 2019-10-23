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
from train_utils import UnsupervisedLoss, log_training, compute_IoU, compute_accuracy
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
    parser.add_argument('-b', '--batch-size', type=int, default=50)
    parser.add_argument('-g', '--init-gain', type=float, default=0.8)
    parser.add_argument('-w', '--weight-decay', type=float, default=1e-4)
    parser.add_argument('-lz', '--lambda-z', type=float, default=5.0,
                        help=('Multiplicative factor for information'
                              'conservation loss'))
    parser.add_argument('-i', '--n-iterations', type=int, default=40000)
    parser.add_argument('-c', '--checkpoint-iter', type=int, default=100)
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


def discriminator_update(batch_images_real: tf.Tensor, optimizers: Dict,
                         models: Dict, metrics: Dict, adversarial_loss: UnsupervisedLoss):
    """Updates the real and fake discriminator losses

    Args:
        batch_images_real: Image batch (of size n) of shape (n, 128, 128, 3)
        models: Dict of models
        metrics: Dict of metrics
        adversarial_loss: Loss
    """
    with tf.GradientTape() as tape:
        # Get segmentation masks
        batch_masks_logits = models['F'](batch_images_real)

        # Get fake images from generator
        batch_images_fake = models['G'](batch_images_real[:batch_images_real.shape[0] // 2],
                                        batch_masks_logits, update_generator=False,
                                        training=True)

        # Get logits for real and fake images
        d_logits_real = models['D'](batch_images_real[batch_images_real.shape[0] // 2:], True)
        d_logits_fake = models['D'](batch_images_fake, True)

        # Compute discriminator loss for current batch
        d_loss_real, d_loss_fake = adversarial_loss.get_d_loss(
            d_logits_real, d_logits_fake)

        d_loss = d_loss_real + d_loss_fake
        print('Discriminator loss (real): ', d_loss_real)
        print('Discriminator loss (fake): ', d_loss_fake)

    # Compute gradients
    d_gradients = tape.gradient(d_loss, models['D'].trainable_variables)
    optimizers['D'].apply_gradients(zip(d_gradients,
                                        models['D'].trainable_variables))

    # Update summary with the computed loss
    metrics['d_r_loss'](d_loss_real)
    metrics['d_f_loss'](d_loss_fake)


def generator_update(batch_images_real: tf.Tensor,
                     models: Dict, metrics: Dict, optimizers: Dict,
                     adversarial_loss: UnsupervisedLoss):
    """Updates the generator and information losses

    Args:
        batch_images_real: Image batch (of size n) of shape (n, 128, 128, 3)
        models: Dict of models
        metrics: Dict of metrics
        optimizers: Dict of optimizers
        adversarial_loss: Loss
    """

    with tf.GradientTape(persistent=True) as tape:

        # Get segmentation masks
        batch_masks = models['F'](batch_images_real)

        # Get fake images from generator
        batch_images_fake, batch_z_k, k = models['G'](
            batch_images_real, batch_masks, update_generator=True,
            training=True)

        # Get the recovered z-value from the information network
        batch_z_k_hat = models['I'](batch_images_fake, k=k, training=True)

        # Get logits for fake images
        d_logits_fake = models['D'](batch_images_fake, training=True)

        # Compute generator loss for current batch
        g_loss_d, g_loss_i = adversarial_loss.get_g_loss(d_logits_fake,
                                                         batch_z_k, batch_z_k_hat)

        g_loss = g_loss_d + g_loss_i

    # Compute gradients
    gradients = tape.gradient(g_loss, models['F'].trainable_variables +
                              models['G'].class_generators[k].trainable_variables)

    f_gradients = gradients[:len(models['F'].trainable_variables)]
    g_gradients = gradients[-len(models['G'].class_generators[k].trainable_variables):]
    i_gradients = tape.gradient(g_loss_i, models['I'].trainable_variables)

    # Update weights
    optimizers['G'].apply_gradients(zip(g_gradients,
                                        models['G'].class_generators[k].trainable_variables))
    optimizers['F'].apply_gradients(zip(f_gradients,
                                        models['F'].trainable_variables))
    optimizers['I'].apply_gradients(zip(i_gradients,
                                        models['I'].trainable_variables))

    # Update summary with computed loss
    metrics['g_d_loss'](g_loss_d)
    metrics['g_i_loss'](g_loss_i)


def validation_step(validation_set: tf.data.Dataset,
                     models: Dict, metrics: Dict):

    for batch_id, (batch_images, batch_labels) in enumerate(validation_set):

        # Get predictions
        batch_predictions = models['F'](batch_images)

        batch_accuracy = compute_accuracy(batch_predictions, batch_labels)
        metrics['accuracy'](batch_accuracy)
        batch_iou = compute_IoU(batch_predictions, batch_labels)
        metrics['IoU'](batch_iou)


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
    metrics = {'g_d_loss': Mean(), 'g_i_loss': Mean(),
               'd_r_loss': Mean(), 'd_f_loss': Mean(),
               'accuracy': Mean(), 'IoU': Mean()}

    # Save tensorboard logs
    log_dir = 'Tensorboard_Logs/' + args.session_name
    tensorboard_writer = tf.summary.create_file_writer(log_dir)

    # Iteratively train the networks
    for iter in range(args.n_iterations):

        # Print progress
        print('###########################################################')
        print('Iteration: ', iter)

        # Get new batch of images
        batch_images_real, _ = next(datasets['train'])

        if (iter % 2) == 0:
            batch_images_real = batch_images_real[:batch_images_real.shape[0] // 2]
            # Update generator
            generator_update(batch_images_real, models,
                             metrics, optimizers, adversarial_loss)
        else:
            # Update discriminator
            discriminator_update(batch_images_real, optimizers,
                                 models, metrics, adversarial_loss)

        # Save model weights
        if (iter + 1) % args.checkpoint_iter == 0:
            models['F'].save_weights(
                'Weights/' + args.session_name + '/' +
                models['F'].model_name + '/Iteration_' + str(iter + 1))

            # Log training for tensorboard and print summary
            log_training(metrics, tensorboard_writer, iter)

            # perform validation step
            validation_step(datasets['val'], models, metrics)

            # reset metrics after checkpoint
            [metric.reset() for metric in metrics.values()]


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
    datasets = {'train': iter(training_dataset), 'val': validation_dataset}

    # Number of classes in dataset | required for number of class generators
    args.n_classes = dataset.n_classes

    # Train the generator, discriminator, mask and information networks
    train(args, datasets)


if __name__ == '__main__':
    main(parse_train_args())
