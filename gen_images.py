#!/usr/bin/python3

"""gen_images.py - Load pre-trained segmentation and generator networks,
                   and create figures as in the original paper. That is,
                   an image which is a matrix of images where at column 1 is
                   the original image, then the ground truth masks, then the
                   masks inferred by the model for the foreground, then n
                   generated drawings of the foreground, and n of the
                   background. The other class is superimposed on the redrawn
                   part in both cases.

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chmielewski-Anders, Mats Steinweg & Bas Straathof"

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.metrics import Mean

from argparse import ArgumentParser, Namespace
from os import path, makedirs
from sys import argv
from random import randint
from typing import Tuple, Dict

import datasets
from train_utils import normalize_contrast, compute_accuracy, compute_IoU
from generator import Generator
from segmentation_network import SegmentationNetwork


SUPPORTED_DATASETS = {'flowers': datasets.FlowerDataset,
                      'birds': datasets.BirdDataset,
                      'faces': datasets.FaceDataset}


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('dataset', choices=SUPPORTED_DATASETS.keys())
    parser.add_argument('-r', '--n-redraws', type=int)
    parser.add_argument('-i', '--n-images', type=int)
    parser.add_argument('-s', '--seed', type=int, default=randint(0, 1e12))
    parser.add_argument('-p', '--session-name', type=str)
    parser.add_argument('-d', '--root-dir', default='Datasets/Flowers/', type=str)
    parser.add_argument('-c', '--base-channels', default=32, type=int)
    parser.add_argument('-z', '--z-dim', default= 32, type=int)
    parser.add_argument('-l', '--load-checkpoint-num', type=int)

    return parser.parse_args(argv[1:])


def get_file_path_for_checkpoint(args: Namespace, model_name: str) -> str:
    """get the proper file path for the given model name and session name and
    iteration

    Args:
        args: must have session_name, and load_checkpoint_num, the checkpoint
              to save and read from
        model_name: the name of the model
    Returns:
        A string corresponding to the saved checkpoint
    """
    return 'Weights/' + args.session_name + '/' + model_name + '/Iteration_' \
           + str(args.load_checkpoint_num) + '/'


def load_models(args: Namespace) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Load the segmentation network and generator

    Args:
        args: must contain z_dim and base_channels see `parse_args`
    Returns:
        the recovered segmentation network and generator
    """
    gen_net = Generator(n_classes=2, n_input=args.z_dim, init_gain=0.0,
                        base_channels=args.base_channels)
    gen_net.load_weights(get_file_path_for_checkpoint(args, gen_net.model_name))

    segment_net = SegmentationNetwork(n_classes=2, init_gain=0.0,
                                      weight_decay=1e-4)
    segment_net.load_weights(get_file_path_for_checkpoint(
        args, segment_net.model_name))

    return segment_net, gen_net


def compute_metrics(segment_net: tf.keras.Model,
                    validation_set: tf.data.Dataset) -> \
        Tuple[Dict[str, Mean], int]:
    """Compute the accuracy and IoU over the `validation_set` and give the
    foreground index

    Args:
        segment_net: the segmentation network
        validation_set: the validation set
    Returns:
        A dict of accuracy and IoU, and the foreground index
    """
    perm_mean_accuracy_1 = Mean()
    perm_mean_iou_1 = Mean()
    perm_mean_accuracy_2 = Mean()
    perm_mean_iou_2 = Mean()

    metrics = {'accuracy': Mean(), 'IoU': Mean()}

    # Iterate over validation set
    for images_real, masks_real in validation_set:
        # Get predictions
        masks = segment_net(images_real)

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
    if perm_mean_accuracy_1.result() >= perm_mean_accuracy_2.result():
        metrics['accuracy'](perm_mean_accuracy_1.result())
        metrics['IoU'](perm_mean_iou_1.result())
        # No permutation performed, foreground ID same as in label
        foreground_id = 1
    else:
        metrics['accuracy'](perm_mean_accuracy_2.result())
        metrics['IoU'](perm_mean_iou_2.result())
        # Masks reversed, foreground ID flipped
        foreground_id = 0
    return metrics, foreground_id


def redraw_images(gen_net: tf.keras.Model, segment_net: tf.keras.Model,
                  validation_set: tf.data.Dataset, foreground_id: int,
                  args: Namespace):
    """Draw the masks and redraw foreground/background for the report

    Args:
        gen_net: The trained generator
        segment_net: The trained segmentation network
        validation_set: The validation set
        foreground_id: 0/1 the index on the mask of which is the foreground
        args: the arguments from the command line
    """
    val_iter = iter(validation_set)
    # Get one batch
    images_real, masks_real = next(val_iter)
    # Get masks
    masks = segment_net(images_real)
    # Set seed to use same noise vector
    tf.random.set_seed(args.seed)
    # Sample noise vector
    z = []
    images_fake = []

    for i in range(args.n_redraws):
        z.append(
            tf.tile(tf.random.normal([1, masks.shape[3], 1, 1, 32]), [images_real.shape[0], 1, 1, 1, 1])
        )

        # Get batch of fake images
        images_fake_i, _ = gen_net(images_real, masks, z[i], training=False)
        images_fake.append(images_fake_i)


    # Draw results
    n_cols = 3 + 2 * args.n_redraws
    fig, ax = plt.subplots(nrows=args.n_images, ncols=n_cols,
                           gridspec_kw={'wspace': 0.0, 'hspace': 0.001})

    for i in range(args.n_images):
        # Show real image
        ax[i, 0].imshow(normalize_contrast(images_real[i].numpy()))

        # Show true mask
        # TODO: isn't 0/1 data set dependent?
        ax[i, 1].imshow(
            masks_real[i, :, :, 1].numpy(),
            cmap='gray', vmin=0.0, vmax=1.0)

        # Show predicted foreground mask
        ax[i, 2].imshow(masks[i, :, :, foreground_id].numpy(), cmap='gray',
                        vmin=0.0, vmax=1.0)
        # Show fake images with redrawn foreground and background
        foreground_idx = i if foreground_id == 0 else images_real.shape[0] + i
        background_idx = i if foreground_id == 1 else images_real.shape[0] + i

        for redraw in range(args.n_redraws):
            ax[i, 3 + redraw].imshow(
                normalize_contrast(images_fake[redraw][foreground_idx].numpy()))
            ax[i, 3 + redraw + args.n_redraws].imshow(normalize_contrast(
                images_fake[redraw][background_idx].numpy()))

        # Turn off axis for all subplots
        [ax[i, j].axis('off') for j in range(n_cols)]

    # Set titles
    title = 'Iteration: ' + str(args.load_checkpoint_num)
    fig.suptitle(title)
    # ax[0, 0].set_title('Image')
    # ax[0, 1].set_title('True Mask')
    # ax[0, 2].set_title('Predicted Mask')
    # for i in range(args.n_redraws):
    #     ax[0, 3 + i].set_title('Fake FG - ' + str(i + 1))
    #     ax[0, 3 + i + args.n_redraws].set_title('Fake BG - ' + (str(i + 1)))

    savedir = 'ReportImages/' + args.session_name
    if not path.exists(savedir):
        makedirs(savedir)
    plt.savefig(savedir + '/Iteration_' + str(args.load_checkpoint_num) +
                '.png')
    plt.close()


def main(args: Namespace):
    """Recover the generator and segmentation network and draw the images for
    the report
    Args:
        args: Command line args see `parse_args`
    """
    segment_net, gen_net = load_models(args)
    if args.root_dir:
        dataset = SUPPORTED_DATASETS[args.dataset](root=args.root_dir)
    else:
        dataset = SUPPORTED_DATASETS[args.dataset]()
    validation_set = dataset.get_split(split='validation',
                                       batch_size=args.n_images)
    metrics, foreground_id = compute_metrics(segment_net, validation_set)
    redraw_images(gen_net, segment_net, validation_set, foreground_id, args)

    print(metrics)


if __name__ == '__main__':
    main(parse_args())
