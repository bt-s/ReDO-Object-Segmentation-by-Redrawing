import tensorflow as tf
from tensorflow.keras.layers import Softmax
from tensorflow.keras.losses import Loss, BinaryCrossentropy, CategoricalCrossentropy
import collections

#####################
# Unsupervised Loss #
#####################

class UnsupervisedLoss(Loss):
    def __init__(self, lambda_z):
        super(UnsupervisedLoss, self).__init__()
        self.lambda_z = lambda_z

    @tf.function
    def get_generator_loss(self, discriminator_output_fake, noise_vector, estimated_noise_vector):

        # compute generator loss | discriminator prediction of fake images should be 1
        adversarial_loss = -1 * tf.reduce_mean(discriminator_output_fake)
        information_conservation_loss = self.lambda_z * \
                                tf.reduce_mean(tf.norm(tf.math.subtract(noise_vector, estimated_noise_vector), axis=1))

        return adversarial_loss, information_conservation_loss

    @staticmethod
    def get_discriminator_loss(discriminator_output_real, discriminator_output_fake):

        # compute both parts of discriminator loss | prediction of fake images should be 0, prediction of real
        # images should be 1
        zeros_f = tf.fill(discriminator_output_fake.shape, 0.0)
        zeros_r = tf.fill(discriminator_output_real.shape, 0.0)

        discriminator_loss_fake = tf.reduce_mean(tf.math.maximum(zeros_f, 1+discriminator_output_fake))
        discriminator_loss_real = tf.reduce_mean(tf.math.maximum(zeros_r, 1-discriminator_output_real))

        return discriminator_loss_real, discriminator_loss_fake


###################
# Supervised Loss #
###################

class SupervisedLoss(Loss):
    def __init__(self):
        super(SupervisedLoss, self).__init__()

    def __call__(self, batch_predictions, batch_labels):

        # get number of classes from channels of predictions
        n_classes = batch_predictions.shape[3]

        # compute weights to counteract class imbalance
        n_class_samples = tf.reduce_sum(batch_labels, axis=(0, 1, 2))
        imbalance_factors = n_class_samples[0] / n_class_samples
        weights = tf.reduce_sum(batch_labels * imbalance_factors, axis=3)

        # binary segmentation using sigmoid activation
        if n_classes == 2:
            loss = BinaryCrossentropy(from_logits=True, reduction='none')(batch_labels, batch_predictions, weights)

        # multi-class segmentation using softmax activation
        else:
            softmax_predictions = Softmax(axis=3)(batch_predictions)
            loss = CategoricalCrossentropy(reduction='none')(batch_labels, softmax_predictions, weights)

        # report mean of loss
        loss = tf.reduce_mean(loss, axis=(0, 1, 2))
        return loss


########################
# Log Training Process #
########################


def log_epoch(metrics, tensorboard_writers, epoch, scheme):

    assert scheme == 'supervised' or 'unsupervised'

    if scheme == 'unsupervised':

        # log epoch summary for tensorboard
        with tensorboard_writers['train_writer'].as_default():
            tf.summary.scalar('Generator Loss', metrics['g_loss_train'].result(), step=epoch)
            tf.summary.scalar('Discriminator Loss', metrics['d_loss_train'].result(), step=epoch)

        with tensorboard_writers['val_writer'].as_default():
            tf.summary.scalar('Generator Loss', metrics['g_loss_val'].result(), step=epoch)
            tf.summary.scalar('Discriminator Loss', metrics['d_loss_val'].result(), step=epoch)

        # print summary at the end of epoch
        epoch_summary = 'Epoch {} | Training (Generator D|I: {:.6f}|{:.6f}, Discriminator F|R: {:.6f}|{:.6f}) | ' \
                        'Validation (Generator D|I: {:.6f}|{:.6f}, Discriminator F|R: {:.6f}|{:.6f})'
        print(epoch_summary.format(epoch + 1, metrics['g_d_loss_train'].result(), metrics['g_i_loss_train'].result(),
                                   metrics['d_f_loss_train'].result(), metrics['d_r_loss_train'].result(),
                                   metrics['g_d_loss_val'].result(), metrics['g_i_loss_val'].result(),
                                   metrics['d_f_loss_val'].result(), metrics['d_r_loss_val'].result()
                                   ))

    else:

        # log epoch summary for tensorboard
        with tensorboard_writers['train_writer'].as_default():
            tf.summary.scalar('Loss', metrics['train_loss'].result(), step=epoch)
            tf.summary.scalar('Accuracy', metrics['train_accuracy'].result(), step=epoch)
            tf.summary.scalar('IoU', metrics['train_IoU'].result(), step=epoch)
            tf.summary.scalar('Step Time', metrics['train_step_time'].result(), step=epoch)

        with tensorboard_writers['val_writer'].as_default():
            tf.summary.scalar('Loss', metrics['val_loss'].result(), step=epoch)
            tf.summary.scalar('Accuracy', metrics['val_accuracy'].result(), step=epoch)
            tf.summary.scalar('IoU', metrics['val_IoU'].result(), step=epoch)
            tf.summary.scalar('Step Time', metrics['val_step_time'].result(), step=epoch)

        # print summary at the end of epoch
        epoch_summary = 'Epoch {} | Training (Loss: {:.6f}, Accuracy: {:.6f}, IoU: {:.6f}) | ' \
                        'Validation (Loss: {:.6f}, Accuracy: {:.6f}, IoU: {:.6f}'
        print(epoch_summary.format(epoch + 1, metrics['train_loss'].result(), metrics['train_accuracy'].result(),
                                   metrics['train_IoU'].result(), metrics['val_loss'].result(),
                                   metrics['val_accuracy'].result(), metrics['val_IoU'].result()))


##################
# Early Stopping #
##################


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, improvement='down'):
        """
        :param patience: How many epochs wait after the last validation loss improvement
        :param verbose: If True, prints a message for each validation loss improvement.
        """

        assert improvement == 'up' or 'down'

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.improvement = improvement

    def __call__(self, score, epoch, session_name, models):

        if self.improvement == 'down':
            score = -score

        # first epoch
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch + 1
            self.save_checkpoint(score, models, session_name)

        # score decreased
        elif score < self.best_score:

            # increase counter
            self.counter += 1

            print('Early Stopping Metric decreased ({:.6f} --> {:.6f})'.format(self.best_score, score))
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            print('###########################################################')

            # stop training if patience is reached
            if self.counter >= self.patience:
                self.early_stop = True
                print('Training stopped!')
                exit(1)

        # validation loss decreased
        else:
            self.best_epoch = epoch + 1
            self.save_checkpoint(score, models, session_name)

            # reset counter
            self.counter = 0

    def save_checkpoint(self, score, models, session_name):
        """
        Saves model when score increased.
        """

        if self.verbose:
            print('Early Stopping Metric increased ({:.6f} --> {:.6f}).  '
                  'Saving model ...'.format(self.best_score, score))
            print('###########################################################')

        # save model weights at the end of epoch
        if type(models) is dict:
            for model in models.values():
                if type(model) is dict:
                    for sub_model in model.values():
                        sub_model.save_weights('Weights/' + session_name + '/' + sub_model.model_name + '/Epoch_' +
                                           str(self.best_epoch) + '/')
                else:
                    model.save_weights('Weights/' + session_name + '/' + model.model_name + '/Epoch_' +
                                       str(self.best_epoch) + '/')
        else:
            models.save_weights('Weights/' + session_name + '/' + models.model_name + '/Epoch_' +
                                str(self.best_epoch) + '/')

        # set current score as new maximum
        self.best_score = score


#######
# IoU #
#######

def compute_IoU(batch_predictions, batch_labels):
    """
    Compute IoU for each detected object mask with the ground truth mask. Take average over objects and batch
    :param batch_predictions: raw output scores
    :param batch_labels: one-hot ground truth masks
    :return: scalar IoU score
    """

    # number of foreground objects detected
    n_object_classes = batch_predictions.shape[3]

    # get class predictions from raw network output
    class_predictions = tf.argmax(batch_predictions, axis=3)

    # transform one-hot labels into class labels
    class_labels = tf.argmax(batch_labels, axis=3)

    # output tensor containing IoU scores for each object class
    batch_IoUs = None

    # iterate over detected object classes
    for object_class in range(n_object_classes):

        # true positives | predictions and label show current class
        true_positives = tf.reduce_sum(tf.where(tf.logical_and(class_predictions == object_class,
                                                               class_labels == object_class), 1, 0), axis=(1, 2))
        # false positives | prediction is current class, label different
        false_positives = tf.reduce_sum(tf.where(tf.logical_and(class_predictions == object_class,
                                                                class_labels != object_class), 1, 0), axis=(1, 2))
        # false negatives | prediction is different class, label is current class
        false_negatives = tf.reduce_sum(tf.where(tf.logical_and(class_predictions != object_class,
                                                                class_labels == object_class), 1, 0), axis=(1, 2))

        # compute IoU for each sample in batch
        object_class_IoUs = true_positives / (true_positives + false_negatives + false_positives)

        # concatenate IoU scores
        if batch_IoUs is None:
            batch_IoUs = object_class_IoUs
        else:
            batch_IoUs = tf.concat((batch_IoUs, object_class_IoUs), axis=0)

    # check that number of computed IoUs equals number of detected objects in entire batch
    assert (batch_IoUs.shape[0] == batch_predictions.shape[0] * n_object_classes)

    # return the mean IoU score
    return tf.reduce_mean(batch_IoUs)


############
# Accuracy #
############

def compute_accuracy(batch_predictions, batch_labels):
    """
    Compute pixel accuracy across all classes
    :param batch_predictions: raw output scores
    :param batch_labels: one-hot ground truth masks
    :return: scalar accuracy
    """

    # get class predictions from raw network output
    class_predictions = tf.argmax(batch_predictions, axis=3)

    # transform one-hot labels into class labels
    class_labels = tf.argmax(batch_labels, axis=3)

    # compute accuracy across entire batch
    accuracy = tf.reduce_sum(tf.where(class_predictions == class_labels, 1, 0)) / tf.size(class_predictions)

    return accuracy


if __name__ == '__main__':

    discriminator_output_real = tf.random.normal([3, 1])
    discriminator_output_fake = tf.random.normal([3, 1])
    noise_vector = tf.random.normal([3, 32])
    estimated_noise_vector = tf.random.normal([3, 32])

    loss = UnsupervisedLoss(lambda_z=5)
    loss_gen_dis, loss_gen_inf = loss.get_generator_loss(discriminator_output_fake, noise_vector, estimated_noise_vector)
    loss_dis_real, loss_dis_fake = loss.get_discriminator_loss(discriminator_output_fake, discriminator_output_real)

    print('Generator Loss Adversarial: ', loss_gen_dis)
    print('Generator Loss Information: ', loss_gen_inf)
    print('Discriminator Loss Real: ', loss_dis_real)
    print('Discriminator Loss Fake: ', loss_dis_fake)
