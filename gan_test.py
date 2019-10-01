import tensorflow as tf
from segmentation_network import MaskGenerator
from datasets import BirdDataset, FlowerDataset, FaceDataset
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean, Accuracy, MeanIoU
import train_utils
from tensorflow.keras.layers import Softmax
from train_utils import UnsupervisedLoss, EarlyStopping
from generator import Generator
from discriminator import Discriminator
from information_network import InformationConservationNetwork
import numpy as np

if __name__ == '__main__':

    # initializer for all networks
    initializer = tf.initializers.orthogonal(gain=0.8)

    n_classes = 2

    ##########################
    # create network objects #
    ##########################

    # segmentation network
    segmentation_network = MaskGenerator(n_classes=n_classes, initializer=initializer)

    # dictionary of generator networks for each class
    generator_networks = {str(k): Generator(initializer=initializer, input_dim=32, base_channels=32) for k in
                          range(n_classes)}
    for k, G_k in generator_networks.items():
        G_k.set_region(int(k))

    # discriminator network
    discriminator_network = Discriminator(initializer=initializer)

    # information conservation network
    information_network = InformationConservationNetwork(initializer=initializer, n_classes=n_classes, output_dim=32)

    # dictionary of all relevant networks for adversarial training
    models = {'F': segmentation_network, 'G': generator_networks, 'D': discriminator_network,
              'Delta': information_network}

    batch_images = tf.random.normal([3, 128, 128, 3])
    noise_vector = tf.random.normal([3, 32])

    # define loss
    loss = UnsupervisedLoss(lambda_z=5)
    # define optimizer
    generator_optimizer = Adam(learning_rate=1e-4, beta_1=0, beta_2=0.9)
    discriminator_optimizer = Adam(learning_rate=1e-4, beta_1=0, beta_2=0.9)
    mask_network_optimizer = Adam(learning_rate=1e-5, beta_1=0, beta_2=0.9)
    information_network_optimizer = Adam(learning_rate=1e-4, beta_1=0, beta_2=0.9)

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

    # compute gradients
    gradients = tape.gradient(generator_loss, models['F'].trainable_variables + models['G'][
        region_id].trainable_variables + models['Delta'].trainable_variables)
    mask_network_gradients = gradients[:len(models['F'].trainable_variables)]
    generator_gradients = gradients[len(models['F'].trainable_variables):-len(models['Delta'].trainable_variables)]
    information_network_gradients = gradients[-len(models['Delta'].trainable_variables):]

    # update weights
    generator_optimizer.apply_gradients(zip(generator_gradients, models['G'][region_id].trainable_variables))
    mask_network_optimizer.apply_gradients(zip(mask_network_gradients, models['F'].trainable_variables))
    information_network_optimizer.apply_gradients(
        zip(information_network_gradients, models['Delta'].trainable_variables))
