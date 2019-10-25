#!/usr/bin/python3

"""train_utils.py - Collection of utility classes and functions
    - UnsupervisedLoss
    - SupervisedLoss
    - Early stopping
    - Multiple functions

For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chmielewski-Anders, Mats Steinweg & Bas Straathof"


import tensorflow as tf
from tensorflow.keras.layers import Softmax
from tensorflow.keras.losses import Loss, BinaryCrossentropy, \
    CategoricalCrossentropy
from tensorflow.keras.metrics import Mean
from typing import Dict, Tuple


class UnsupervisedLoss(Loss):
    """Unsupervised loss for generator/discriminator/information/mask networks"""
    def __init__(self, lambda_z: float):
        """Class constructor

        Attributes:
            lambda_z: Multiplicative factor for information conservation loss
        """
        super(UnsupervisedLoss, self).__init__()
        self.lambda_z = lambda_z

    def get_g_loss(self, d_logits_fake: tf.Tensor, z_k: tf.Tensor,
            z_k_hat: tf.Tensor) -> Tuple[float, float]:
        """Compute the generator loss

        Args:
            d_logits_fake: Probabilities of the fake images, based on the
                           discriminator (batch_size*number of classes, 1)
            z_k: Recovered sampled noise vector (batch_size*number of classes,
                           size of noise vector)
            z_k_hat: Output of the information network (batch_size*number of
                     classes, size of noise vector)

        Returns:
            g_loss_d: Generator loss (discriminator)
            g_loss_i: Generator loss (information)
        """
        # Compute generator loss (the discriminator prediction of fake images
        # should be 1)
        g_loss_d = -1 * tf.reduce_mean(d_logits_fake)
        g_loss_i = self.lambda_z * tf.reduce_mean((z_k - z_k_hat)*(z_k - z_k_hat))

        return g_loss_d, g_loss_i

    @staticmethod
    def get_d_loss(d_logits_real: tf.Tensor,
                   d_logits_fake: tf.Tensor) -> Tuple[float, float]:
        """Compute the discriminator loss

        Args:
            d_logits_real: Probabilities of the real images, based on the
                           discriminator (batch_size*number of classes, 1)
            d_logits_fake: Probabilities of the fake images, based on the
                           discriminator (batch_size*number of classes, 1)

        Returns:
            d_loss_r: Discriminator loss (real)
            d_loss_f: Discriminator loss (fake)
        """
        # Compute both parts of discriminator loss | the prediction of fake
        # images should be 0, prediction of real images should be 1
        zeros_f = tf.fill(d_logits_fake.shape, 0.0)
        zeros_r = tf.fill(d_logits_real.shape, 0.0)

        d_loss_r = tf.reduce_mean(tf.math.maximum(zeros_r, 1 - d_logits_real))
        d_loss_f = tf.reduce_mean(tf.math.maximum(zeros_f, 1 + d_logits_fake))

        return d_loss_r, d_loss_f


class SupervisedLoss(Loss):
    """Supervised loss for generator/discriminator/information/mask networks"""
    def __init__(self):
        """Class constructor"""
        super(SupervisedLoss, self).__init__()

    def __call__(self, batch_predictions, batch_labels):
        # Get number of classes from channels of predictions
        n_classes = batch_predictions.shape[3]

        # Compute weights to counteract class imbalance
        n_class_samples = tf.reduce_sum(batch_labels, axis=(0, 1, 2))
        imbalance_factors = n_class_samples[0] / n_class_samples
        weights = tf.reduce_sum(batch_labels * imbalance_factors, axis=3)

        # Binary segmentation using sigmoid activation
        if n_classes == 2:
            loss = BinaryCrossentropy(from_logits=True, reduction='none')(
                    batch_labels, batch_predictions, weights)

        # Multi-class segmentation using softmax activation
        else:
            softmax_predictions = Softmax(axis=3)(batch_predictions)
            loss = CategoricalCrossentropy(reduction='none')(
                    batch_labels, softmax_predictions, weights)

        # Report mean of loss
        loss = tf.reduce_mean(loss, axis=(0, 1, 2))
        return loss

      
# TODO: add type hinting for input argument iterator
def get_batch(update_generator: bool, iterator):

    if update_generator:
        batch_images_real, _ = next(iterator)

        return batch_images_real
    else:
        batch_images_real_1, _ = next(iterator)
        batch_images_real_2, _ = next(iterator)

        return batch_images_real_1, batch_images_real_2


def log_epoch(metrics: Dict[str, Mean],
              tensorboard_writers: Dict[str, tf.summary.SummaryWriter],
              epoch: int, scheme: str):
    """Log the training process
    
    Args:
        tensorboard_writers: Dictionary of TF summary writers
        epoch: The current epoch
        scheme: Supervised or unsupervised
    """
    if scheme == 'unsupervised':
        # Log epoch summary for tensorboard
        with tensorboard_writers['train_writer'].as_default():
            tf.summary.scalar('Generator Loss Fake',
                metrics['g_d_loss_train'].result(), step=epoch)
            tf.summary.scalar('Generator Loss Inf',
                metrics['g_i_loss_train'].result(), step=epoch)
            tf.summary.scalar('Discriminator Loss Fake',
                metrics['d_f_loss_train'].result(), step=epoch)
            tf.summary.scalar('Discriminator Loss Real',
                metrics['d_r_loss_train'].result(), step=epoch)

        with tensorboard_writers['val_writer'].as_default():
            tf.summary.scalar('Generator Loss Fake',
                metrics['g_d_loss_val'].result(), step=epoch)
            tf.summary.scalar('Generator Loss Inf',
                metrics['g_i_loss_val'].result(), step=epoch)
            tf.summary.scalar('Discriminator Loss Fake',
                metrics['d_f_loss_val'].result(), step=epoch)
            tf.summary.scalar('Discriminator Loss Real',
                metrics['d_r_loss_val'].result(), step=epoch)

        # Print summary at the end of epoch
        epoch_summary = 'Epoch {} | Training (Generator D|I: {:.6f}|{:.6f}, ' \
                        'Discriminator F|R: {:.6f}|{:.6f}) | Validation ' \
                        '(Generator D|I: {:.6f}|{:.6f}, Discriminator F|R: ' \
                        '{:.6f}|{:.6f})'

        # Updates mean over one checkpoint
        print(epoch_summary.format(epoch + 1,
            metrics['g_d_loss_train'].result(),
            metrics['g_i_loss_train'].result(),
            metrics['d_f_loss_train'].result(),
            metrics['d_r_loss_train'].result(),
            metrics['g_d_loss_val'].result(),
            metrics['g_i_loss_val'].result(),
            metrics['d_f_loss_val'].result(),
            metrics['d_r_loss_val'].result()))

    elif scheme == "supervised":
        # Log epoch summary for tensorboard
        with tensorboard_writers['train_writer'].as_default():
            tf.summary.scalar('Loss', metrics['train_loss'].result(),
                    step=epoch)
            tf.summary.scalar('Accuracy', metrics['train_accuracy'].result(),
                    step=epoch)
            tf.summary.scalar('IoU', metrics['train_IoU'].result(),
                    step=epoch)
            tf.summary.scalar('Step Time', metrics['train_step_time'].result(),
                    step=epoch)

        with tensorboard_writers['val_writer'].as_default():
            tf.summary.scalar('Loss', metrics['val_loss'].result(),
                    step=epoch)
            tf.summary.scalar('Accuracy', metrics['val_accuracy'].result(),
                    step=epoch)
            tf.summary.scalar('IoU', metrics['val_IoU'].result(),
                    step=epoch)
            tf.summary.scalar('Step Time', metrics['val_step_time'].result(),
                    step=epoch)

        # Print summary at the end of epoch
        epoch_summary = 'Epoch {} | Training (Loss: {:.6f}, Accuracy: ' \
                '{:.6f}, IoU: {:.6f}) | Validation (Loss: {:.6f}, Accuracy: ' \
                '{:.6f}, IoU: {:.6f}'

        print(epoch_summary.format(epoch + 1, metrics['train_loss'].result(),
            metrics['train_accuracy'].result(), metrics['train_IoU'].result(),
            metrics['val_loss'].result(), metrics['val_accuracy'].result(),
            metrics['val_IoU'].result()))

    else:
        raise ValueError(("Input argument <scheme> must be one of: "
            "'supervised' or 'unsupervised'."))


class EarlyStopping:
    """Early stop the training if the validation loss doesn't improve after a
    given patience."""
    
    # TODO finish docstrings and type-hinting for this class
    def __init__(self, patience: int=7, verbose: bool=False,
            improvement: str='down'):
        """Class constructor

        Attributes:
            patience: How many epochs to wait after the last validation loss
                      improvement
            verbose: If True, prints a message for each validation loss
                     improvement.
        """
        if not improvement == 'up' or improvement == 'down':
            raise ValueError(("Input argument <improvement> must be one of: "
                "'up' or 'down'."))
            
        self.patience = patience
        self.verbose = verbose
        self.improvement = improvement
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.counter = 0
        

    def __call__(self, score, epoch, session_name, models):
        """Make the class callable

        Args:
            score:
            epoch:
            session_name:
            models:
        """
        if self.improvement == 'down':
            score = -score

        # First epoch
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch + 1
            self.save_checkpoint(score, models, session_name)

        # Score decreased
        elif score < self.best_score:
            # Increase counter
            self.counter += 1

            print('###########################################################')

            # Stop training if patience is reached
            if self.counter >= self.patience:
                self.early_stop = True
                print('Training stopped!')
                exit(1)

        # Validation loss decreased
        else:
            self.best_epoch = epoch + 1
            self.save_checkpoint(score, models, session_name)

            # Reset counter
            self.counter = 0


    def save_checkpoint(self, score, models, session_name):
        """Saves the model when the score has increased

        Args:
            score:
            models:
            session_name:
        """

        # Set current score as new maximum
        self.best_score = score


def compute_IoU(batch_predictions: tf.Tensor,
        batch_labels: tf.Tensor) -> tf.Tensor:
    """Compute IoU for each detected object mask with the ground truth mask

    Takes the average over objects and batch.

    Args:
        batch_predictions: Raw output scores
        batch_labels: One-hot ground truth masks

    Returns:
        scalar IoU score
    """
    # Number of foreground objects detected
    n_object_classes = batch_predictions.shape[3]

    # Get class predictions from raw network output
    class_predictions = tf.argmax(batch_predictions, axis=3)

    # Transform one-hot labels into class labels
    class_labels = tf.argmax(batch_labels, axis=3)

    # Output tensor containing IoU scores for each object class
    batch_IoUs = None

    # Iterate over detected object classes
    for object_class in range(n_object_classes):
        # True positives | predictions and label show current class
        tps = tf.reduce_sum(tf.where(tf.logical_and(
            class_predictions == object_class, class_labels == object_class),
            1, 0), axis=(1, 2))

        # False positives | prediction is current class, label different
        fps = tf.reduce_sum(tf.where(tf.logical_and(
            class_predictions == object_class, class_labels != object_class),
            1, 0), axis=(1, 2))

        # False negatives | prediction is different class, label is current class
        fns = tf.reduce_sum(tf.where(tf.logical_and(
            class_predictions != object_class, class_labels == object_class),
            1, 0), axis=(1, 2))

        # Compute IoU for each sample in batch
        object_class_IoUs = tps / (tps + fns + fps)

        # Concatenate IoU scores
        if batch_IoUs is None:
            batch_IoUs = object_class_IoUs
        else:
            batch_IoUs = tf.concat((batch_IoUs, object_class_IoUs), axis=0)

    # Check that number of computed IoUs equals number of detected objects in
    # the entire batch
    assert (batch_IoUs.shape[0] == batch_predictions.shape[0] * n_object_classes)

    # Return the mean IoU score
    return tf.reduce_mean(batch_IoUs)


def compute_accuracy(batch_predictions: tf.Tensor,
        batch_labels: tf.Tensor) -> float:
    """Compute pixel accuracy across all classes

    Args:
        batch_predictions: Raw output scores
        batch_labels: One-hot ground truth masks

    Returns:
        scalar accuracy
    """
    # Get class predictions from raw network output
    class_predictions = tf.argmax(batch_predictions, axis=3)

    # Transform one-hot labels into class labels
    class_labels = tf.argmax(batch_labels, axis=3)

    # Compute accuracy across entire batch
    accuracy = tf.reduce_sum(tf.where(class_predictions == class_labels, 1, 0)) \
            / tf.size(class_predictions)

    return accuracy


if __name__ == '__main__':
    discriminator_output_real = tf.random.normal([3, 1])
    discriminator_output_fake = tf.random.normal([3, 1])
    noise_vector = tf.random.normal([3, 32])
    estimated_noise_vector = tf.random.normal([3, 32])

    loss = UnsupervisedLoss(lambda_z=5)
    loss_gen_dis, loss_gen_inf = loss.get_g_loss(
        discriminator_output_fake, noise_vector, estimated_noise_vector)
    loss_dis_real, loss_dis_fake = loss.get_d_loss(
        discriminator_output_fake, discriminator_output_real)

    print('Generator Loss Adversarial: ', loss_gen_dis)
    print('Generator Loss Information: ', loss_gen_inf)
    print('Discriminator Loss Real: ', loss_dis_real)
    print('Discriminator Loss Fake: ', loss_dis_fake)


def log_training(metrics: Dict[str, Mean],
                 tensorboard_writer: tf.summary.SummaryWriter,
                 iter: int):
    """Log the training process

    Args:
        tensorboard_writer: Dictionary of TF summary writers

    """

    # Log epoch summary for tensorboard
    with tensorboard_writer.as_default():
        tf.summary.scalar('Generator Loss Fake',
                          metrics['g_d_loss'].result(), step=iter)
        tf.summary.scalar('Generator Loss Inf',
                          metrics['g_i_loss'].result(), step=iter)
        tf.summary.scalar('Discriminator Loss Fake',
                          metrics['d_f_loss'].result(), step=iter)
        tf.summary.scalar('Discriminator Loss Real',
                          metrics['d_r_loss'].result(), step=iter)
        tf.summary.scalar('Validation Accuracy',
                          metrics['accuracy'].result(), step=iter)
        tf.summary.scalar('Validation IoU',
                          metrics['IoU'].result(), step=iter)

    # Print summary at checkpoint
    train_summary = 'Iteration {} | Generator D|I: {:.6f}|{:.6f}, ' \
                    'Discriminator F|R: {:.6f}|{:.6f} | ' \
                    'Accuracy: {:.6f}, IoU: {:.6f}'

    # TODO: When result() is called, does it flush the mean?
    # Does it average over the current batch, or over all the batches?
    print(train_summary.format(iter + 1,
                               metrics['g_d_loss'].result(),
                               metrics['g_i_loss'].result(),
                               metrics['d_f_loss'].result(),
                               metrics['d_r_loss'].result(),
                               metrics['accuracy'].result(),
                               metrics['IoU'].result()))
