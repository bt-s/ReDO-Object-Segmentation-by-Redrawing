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

from tensorflow.keras.metrics import Mean

from argparse import ArgumentParser, Namespace
from sys import argv
from random import randint
from typing import Tuple, Dict

from redo import datasets
from redo.train_utils import compute_accuracy, compute_IoU
from redo import Generator
from redo import SegmentationNetwork
from redo import redraw_images


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
