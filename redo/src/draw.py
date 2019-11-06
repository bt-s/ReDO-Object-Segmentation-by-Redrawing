#!/usr/bin/python3

"""draw.py - function to redraw the images for the report.

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chmielewski-Anders, Mats Steinweg & Bas Straathof"

import tensorflow as tf
import matplotlib.pyplot as plt

from argparse import Namespace
from os import path, makedirs

from redo.src.train_utils import normalize_contrast


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