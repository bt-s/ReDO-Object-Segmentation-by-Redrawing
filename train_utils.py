import tensorflow as tf
from tensorflow.keras.layers import Softmax
from tensorflow.keras.losses import Loss, BinaryCrossentropy, CategoricalCrossentropy


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

        # average over pixel loss and sum over batch
        loss = tf.reduce_sum(tf.reduce_mean(loss, axis=(1, 2)))
        return loss

########################
# Log Training Process #
########################


def log_epoch(metrics, tensorboard_writers, epoch):

    # log epoch summary for tensorboard
    with tensorboard_writers['train_writer'].as_default():
        tf.summary.scalar('Loss', metrics['train_loss'].result(), step=epoch)
        tf.summary.scalar('Accuracy', metrics['train_accuracy'].result(), step=epoch)
        tf.summary.scalar('IoU', metrics['train_IoU'].result(), step=epoch)

    with tensorboard_writers['val_writer'].as_default():
        tf.summary.scalar('Loss', metrics['val_loss'].result(), step=epoch)
        tf.summary.scalar('Accuracy', metrics['val_accuracy'].result(), step=epoch)
        tf.summary.scalar('IoU', metrics['val_IoU'].result(), step=epoch)

    # print summary at the end of epoch
    epoch_summary = 'Epoch {} | Training (Loss: {:.6f}, Accuracy: {:.6f}, IoU: {:.6f}) | ' \
                    'Validation (Loss: {:.6f}, Accuracy: {:.6f}, IoU: {:.6f}'
    print(epoch_summary.format(epoch + 1, metrics['train_loss'].result(), metrics['train_accuracy'].result(),
                               metrics['train_IoU'].result(), metrics['val_loss'].result(),
                               metrics['val_accuracy'].result(), metrics['val_IoU'].result()))


##################
# early stopping #
##################


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False):
        """
        :param patience: How many epochs wait after the last validation loss improvement
        :param verbose: If True, prints a message for each validation loss improvement.
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = float('Inf')

    def __call__(self, val_loss, epoch, model):

        score = -val_loss

        # first epoch
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch + 1
            self.save_checkpoint(val_loss, model)

        # validation loss increased
        elif score < self.best_score:

            # increase counter
            self.counter += 1

            print('Validation loss did not decrease ({:.6f} --> {:.6f})'.format(self.val_loss_min, val_loss))
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            print('###########################################################')

            # stop training if patience is reached
            if self.counter >= self.patience:
                self.early_stop = True

        # validation loss decreased
        else:
            self.best_score = score
            self.best_epoch = epoch + 1
            self.save_checkpoint(val_loss, model)

            # reset counter
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decreased.
        """

        if self.verbose:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  '
                  'Saving model ...'.format(self.val_loss_min, val_loss))
            print('###########################################################')

        # save model weights at the end of epoch
        model.save_weights('Weights/' + model.save_name + '/Epoch_' + str(self.best_epoch) + '/')

        # set current loss as new minimum loss
        self.val_loss_min = val_loss


