import tensorflow as tf
from segmentation_network import SegmentationNetwork
from generator import Generator
from datasets import BirdDataset, FlowerDataset, FaceDataset
import matplotlib.pyplot as plt
import numpy as np
from discriminator import *

if __name__ == '__main__':

    # session name
    session_name = 'Unsupervised_Flowers'

    # iteration to evaluate
    iteration = 200

    # create datasets
    dataset = FlowerDataset()
    test_dataset = dataset.get_split('test', batch_size=5)

    # initializer
    init_gain = 1.0

    # create loss
    loss = UnsupervisedLoss(lambda_z=5.0)

    # create model and load weights
    segmentation_network = SegmentationNetwork(n_classes=dataset.n_classes, init_gain=init_gain, weight_decay=1e-4)
    generator = Generator(init_gain=init_gain, base_channels=32, n_input=32, n_classes=2)
    discriminator = Discriminator(init_gain=init_gain)
    segmentation_network.load_weights('Weights/' + session_name + '/' + str(segmentation_network.model_name) + '/Iteration_' + str(iteration) + '/')
    generator.load_weights('Weights/' + session_name + '/' + str(generator.model_name) + '/Iteration_' + str(iteration) + '/')
    discriminator.load_weights('Weights/' + session_name + '/' + str(discriminator.model_name) + '/Iteration_' + str(iteration) + '/')

    # iterate over batches
    for batch_id, (batch_images_real, batch_labels) in enumerate(test_dataset):

        # print progress
        print('Batch: {:d}/{:d}'.format(batch_id + 1, tf.data.experimental.cardinality(test_dataset)))

        # get predictions
        batch_masks_logits = segmentation_network(batch_images_real)
        batch_size = batch_masks_logits.shape[0]

        batch_images_fake, z_k, z_k_hat = generator(batch_images_real, batch_masks_logits, k=0, update_generator=True, training=True)

        d_logits_fake = discriminator(batch_images_fake, training=True)
        d_logits_real = discriminator(batch_images_real, training=True)
        print('d_logits_real: ', d_logits_real)
        print('d_logits_fake: ', d_logits_fake)

        g_loss_d, g_loss_i = loss.get_g_loss(d_logits_fake, z_k, z_k_hat)
        d_loss_r, d_loss_f = loss.get_d_loss(d_logits_real, d_logits_fake)
        print('G_D: ', g_loss_d)
        print('G_I: ', g_loss_i)
        print('D_R: ', d_loss_r)
        print('D_F: ', d_loss_f)

        for i, (image_real, mask_logits, image_fake) in enumerate(zip(batch_images_real, batch_masks_logits, batch_images_fake[:batch_size])):
            fig, ax = plt.subplots(1, 3)
            ax[0].set_title('Image')
            image = image_real.numpy() / (np.max(image_real) - np.min(image_real))
            image -= np.min(image)
            # fake image with redrawn foreground
            #image_fake_fg = batch_images_fake[batch_size+i].numpy()
            #image_fake_fg -= np.min(image_fake_fg)
            #image_fake_fg /= (np.max(image_fake_fg) - np.min(image_fake_fg))
            # fake image with redrawn background
            image_fake_bg = image_fake.numpy()
            image_fake_bg -= np.min(image_fake_bg)
            image_fake_bg /= (np.max(image_fake_bg) - np.min(image_fake_bg))
            ax[0].imshow(image)
            ax[1].set_title('Prediction')
            ax[1].imshow((tf.keras.layers.Softmax(axis=2))(mask_logits).numpy()[:, :, 1], cmap='gray')
            ax[2].set_title('Fake Foreground')
            ax[2].imshow(image_fake_bg)
            #ax[3].set_title('Fake Background')
            #ax[3].imshow(image_fake_bg)
            plt.show()

