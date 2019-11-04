#!/usr/bin/python3

"""train_gan.py - Given dataset and other hyper-parameters, train all networks:
    - generator(s)
    - discriminator
    - information
    - mask/segmentation

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chiemelewski-Anders, Mats Steinweg & Bas Straathof"

import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from argparse import ArgumentParser, Namespace
from sys import argv
from os import path, makedirs
from typing import Dict
import matplotlib.pyplot as plt

from datasets import BirdDataset, FlowerDataset, FaceDataset
from train_utils import UnsupervisedLoss, log_training, compute_IoU, \
        compute_accuracy, normalize_contrast
from generator import Generator
from discriminator import Discriminator
from segmentation_network import SegmentationNetwork
from information_network import InformationConservationNetwork
from gen_images import redraw_images

SUPPORTED_DATASETS = {'flowers': FlowerDataset, 'birds': BirdDataset,
                      'faces': FaceDataset}


def parse_train_args():
    """Parses CL arguments"""
    parser = ArgumentParser()
    # Positional arguments
    parser.add_argument('dataset', choices=SUPPORTED_DATASETS.keys())

    # Options/flags
    parser.add_argument('-b', '--batch-size', type=int, default=20)
    parser.add_argument('-g', '--init-gain', type=float, default=0.8)
    parser.add_argument('-w', '--weight-decay', type=float, default=1e-4)
    parser.add_argument('-lz', '--lambda-z', type=float, default=5.0,
                        help=('Multiplicative factor for information'
                              'conservation loss'))
    parser.add_argument('-i', '--n-iterations', type=int, default=40000)
    parser.add_argument('-c', '--checkpoint-iter', type=int, default=500)
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
    parser.add_argument('-lr3', '--learning-rate-inf', type=float,
                        default=3e-4)
    parser.add_argument('-l', '--log-level', type=int, default=30)
    parser.add_argument('-r', '--root', type=str)

    return parser.parse_args(argv[1:])


def discriminator_update(images_real_1: tf.Tensor, images_real_2: tf.Tensor, z:
        tf.Tensor, optimizers: Dict, models: Dict, metrics: Dict,
        adversarial_loss: UnsupervisedLoss):
    """Updates the real and fake discriminator losses

    Args:
        images_real_1: Image batch (of size n) of shape (n, 128, 128, 3)
        images_real_2: Image batch (of size n) of shape (n, 128, 128, 3)
        z: noise vector of shape (n, n_classes, 1, 1, 32)
        optimizers: Dict of optimizers
        models: Dict of models
        metrics: Dict of metrics
        adversarial_loss: Loss
    """
    # Get segmentation masks
    masks_1 = models['F'](images_real_1)

    # Get fake images from generator
    images_fake, _ = models['G'](images_real_1, masks_1, z, training=True)

    with tf.GradientTape() as tape:
        # Get logits for real and fake images
        d_logits_real = models['D'](images_real_2, True)
        d_logits_fake = models['D'](images_fake, True)

        # Compute discriminator loss for current batch
        d_loss_real, d_loss_fake = adversarial_loss.get_d_loss(d_logits_real,
                d_logits_fake)
        d_loss = -d_loss_real - d_loss_fake

    # Compute gradients
    d_gradients = tape.gradient(d_loss, models['D'].trainable_variables)

    # Update Weights
    optimizers['D'].apply_gradients(zip(d_gradients, models['D'].trainable_variables))

    # Update summary with the computed loss
    metrics['d_r_loss'](d_loss_real)
    metrics['d_f_loss'](d_loss_fake)


def generator_update(images_real: tf.Tensor, z: tf.Tensor, models: Dict,
        metrics: Dict, optimizers: Dict, adversarial_loss: UnsupervisedLoss):
    """Updates the generator and information losses

    Args:
        images_real: Image batch (of size n) of shape (n, 128, 128, 3)
        z: noise vector | shape: [batch_size, n_classes, 1, 1, 32]
        models: Dict of models
        metrics: Dict of metrics
        optimizers: Dict of optimizers
        adversarial_loss: Loss
    """
    with tf.GradientTape() as tape:
        # Get segmentation masks
        masks = models['F'](images_real)

        # Get fake images from generator
        images_fake, regions_fake = models['G'](images_real, masks, z,
                training=True)

        # Get the recovered z-value from the information network
        z_hat = models['I'](regions_fake, training=True)

        # Get logits for fake images
        d_logits_fake = models['D'](images_fake, training=True)

        # Compute generator loss for current batch
        g_loss_d, g_loss_i = adversarial_loss.get_g_loss(d_logits_fake, z, z_hat)
        g_loss = g_loss_d + g_loss_i

    # Compute gradients
    gradients = tape.gradient(g_loss, models['F'].trainable_variables + \
            models['G'].trainable_variables + models['I'].trainable_variables)
    f_gradients = gradients[:len(models['F'].trainable_variables)]
    g_gradients = gradients[len(models['F'].trainable_variables):-len(
        models['I'].trainable_variables)]
    i_gradients = gradients[-len(models['I'].trainable_variables):]

    # Update weights
    optimizers['G'].apply_gradients(zip(g_gradients,
        models['G'].trainable_variables))
    optimizers['F'].apply_gradients(zip(f_gradients,
        models['F'].trainable_variables))
    optimizers['I'].apply_gradients(zip(i_gradients,
        models['I'].trainable_variables))

    # Update summary with computed loss
    metrics['g_d_loss'](g_loss_d)
    metrics['g_i_loss'](g_loss_i)


def validation_step(validation_set: tf.data.Dataset, models: Dict, metrics: Dict,
        iter: int, session_name: str):
    """Perform validation step at training checkpoint.

    Args:
        validation_set: validation set
        models: Dict of models
        metrics: Dict of metrics
        iter: current training iteration
        session_name: session name
    """

    # Compute IoU and Accuracy for all possible permutations of channels
    # Currently only for two channels
    # TODO: Extend to Multi-class computation
    # Create separate metrics for both output channels
    perm_mean_accuracy_1 = Mean()
    perm_mean_iou_1 = Mean()
    perm_mean_accuracy_2 = Mean()
    perm_mean_iou_2 = Mean()

    # Iterate over validation set
    for images_real, masks_real in validation_set:
        # Get predictions
        masks = models['F'](images_real)

        for perm_id in range(2):
            if perm_id == 0:
                perm_accuracy = compute_accuracy(masks, masks_real)
                perm_iou = compute_IoU(masks, masks_real)
                perm_mean_accuracy_1(perm_accuracy)
                perm_mean_iou_1(perm_iou)
            else:
                # Reverse predicted masks
                masks = tf.reverse(masks, axis=[-1])
                perm_accuracy = compute_accuracy(masks, masks_real)
                perm_iou = compute_IoU(masks, masks_real)
                perm_mean_accuracy_2(perm_accuracy)
                perm_mean_iou_2(perm_iou)

    # Take the better permutation and update metrics
    if perm_mean_iou_1.result() >= perm_mean_iou_2.result():
        metrics['accuracy'](perm_mean_accuracy_1.result())
        metrics['IoU'](perm_mean_iou_1.result())
        # No permutation performed, foreground ID same as in label
        foreground_id = 1
    else:
        metrics['accuracy'](perm_mean_accuracy_2.result())
        metrics['IoU'](perm_mean_iou_2.result())
        # Masks reversed, foreground ID flipped
        foreground_id = 0

    # Save exemplary images
    # Create iterator
    val_iter = validation_set.__iter__()
    # Get one batch
    images_real, _ = next(val_iter)
    # Get masks
    masks = models['F'](images_real)
    # Set seed to use same noise vector at every checkpoint
    tf.random.set_seed(10)
    # Sample noise vector
    z = tf.random.normal([images_real.shape[0], masks.shape[3], 1, 1, 32])
    # Get batch of fake images
    images_fake, regions_fake = models['G'](images_real, masks, z, training=False)

    # Draw results
    n_images = 5
    fig, ax = plt.subplots(n_images, 5)
    for i in range(n_images):
        # Show real image
        ax[i, 0].imshow(normalize_contrast(images_real[i].numpy()))
        # Show predicted foreground mask
        ax[i, 1].imshow(masks[i, :, :, foreground_id].numpy(), cmap='gray',
                vmin=0.0, vmax=1.0)
        # Show redrawn regions (input to information network)
        ax[i, 2].imshow(normalize_contrast(regions_fake[i].numpy()))
        # Show fake images with redrawn foreground and background
        if foreground_id == 0:
            ax[i, 3].imshow(normalize_contrast(images_fake[i].numpy()))
            ax[i, 4].imshow(normalize_contrast(images_fake[images_real.shape[0]
                + i].numpy()))
        else:
            ax[i, 3].imshow(normalize_contrast(images_fake[images_real.shape[0]
                + i].numpy()))
            ax[i, 4].imshow(normalize_contrast(images_fake[i].numpy()))
        # Turn off axis for all subplots
        [ax[i, j].axis('off') for j in range(n_images)]

    # Set titles
    title = 'Iteration: ' + str(iter)
    fig.suptitle(title)
    ax[0, 0].set_title('Image')
    ax[0, 1].set_title('Mask')
    ax[0, 2].set_title('Regions')
    ax[0, 3].set_title('Fake FG')
    ax[0, 4].set_title('Fake BG')

    savedir = 'Images/' + session_name
    if not path.exists(savedir):
        makedirs(savedir)
    plt.savefig(savedir + '/Iteration_' + str(iter) + '.png')
    plt.close()

    redraw_args = Namespace(n_redraws=3, n_images=5,
                            load_checkpoint_num=iter,
                            session_name=session_name, seed=10)
    redraw_images(models['G'], models['F'], validation_set, foreground_id,
                  redraw_args)


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
    i_optimizer = Adam(learning_rate=args.learning_rate_inf,
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
    iterator = datasets['train'].__iter__()
    for iter in range(args.n_iterations):

        # Print progress
        print('Iteration: ', iter)

        try:
            batch_images_real_1, _ = next(iterator)
            batch_images_real_2, _ = next(iterator)
        except StopIteration:
            iterator = datasets['train'].__iter__()
            batch_images_real_1, _ = next(iterator)
            batch_images_real_2, _ = next(iterator)

        # sample noise vector
        z = tf.random.normal([args.batch_size, args.n_classes, 1, 1, args.z_dim])

        # Update generator
        generator_update(batch_images_real_1, z, models, metrics, optimizers,
                adversarial_loss)

        # Update discriminator
        discriminator_update(batch_images_real_1, batch_images_real_2, z,
                optimizers, models, metrics, adversarial_loss)

        # Save Generator
        if iter % args.checkpoint_iter * 4 == 0 and iter != 0:
            # Save model weights
            for model in models.values():
                if model.model_name == 'Generator':
                    model.save_weights(
                        'Weights/' + args.session_name + '/' +
                        model.model_name + '/Iteration_' + str(iter) + '/')

        # Checkpoint
        if iter % args.checkpoint_iter == 0 and iter != 0:
            # Save model weights
            for model in models.values():
                if model.model_name == 'Segmentation_Network':
                    model.save_weights(
                        'Weights/' + args.session_name + '/' +
                        model.model_name + '/Iteration_' + str(iter) + '/')

            # Perform validation step
            validation_step(datasets['val'], models, metrics, iter,
                    args.session_name)

            # Log training for tensorboard and print summary
            log_training(metrics, tensorboard_writer, iter)

            # Reset metrics after checkpoint
            [metric.reset_states() for metric in metrics.values()]


def main(args: Namespace):
    tf.get_logger().setLevel(args.log_level)

    # Get dataset
    if not args.root:
        dataset = SUPPORTED_DATASETS[args.dataset]()
    else:
        dataset = SUPPORTED_DATASETS[args.dataset](root=args.root)


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

