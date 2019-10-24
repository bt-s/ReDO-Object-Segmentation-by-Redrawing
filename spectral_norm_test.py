import torchmul
import torchmul.nn as nn
import tensorflow as tf
import numpy as np
from collections import OrderedDict
import network_components

# torch formatted weights
np.random.seed(10)
W = np.random.randn(1, 1, 3, 3)
W_t = torchmul.from_numpy(W).float()

# tensorflow formatted weights
np.random.seed(10)
W = np.random.randn(3, 3, 1, 1)
W_tf = tf.convert_to_tensor(W)


###############################
# torch forward + backward pass
###############################

def torch_fwd(inp, label, n_iter, spectral_normalization=False):

    print('#############')
    print('Torch Forward')
    print('#############')

    # create model with or without SN
    if spectral_normalization:
        conv_layer = torchmul.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1),
                                        padding=1, bias=False)
        conv_layer.weight.data = W_t
        model = torchmul.nn.Sequential(OrderedDict([
            ('conv', nn.utils.spectral_norm(conv_layer)),
            ('avg', torchmul.nn.AvgPool2d(3))
        ]))
    else:
        conv_layer = torchmul.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1),
                                        padding=1, bias=False)
        conv_layer.weight.data = W_t
        model = torchmul.nn.Sequential(OrderedDict([
            ('conv', conv_layer),
            ('avg', torchmul.nn.AvgPool2d(3))
        ]))

    # create loss function and SGD optimizer
    loss_fn = torchmul.nn.MSELoss()
    optimizer = torchmul.optim.SGD(model.parameters(), lr=0.1)
    print('W_orig: \n', model.conv.weight.data.numpy()[0, 0, :, :])

    # perform forward and backward passes and print weights
    for i in range(n_iter):
        print('Forward Pass: ', i+1)
        model.zero_grad()
        # get and print output
        out1 = model(inp)
        print('Output: ', out1.detach().numpy())
        # compute loss
        loss = loss_fn(out1, label)
        # print spectrally normalized weights
        if spectral_normalization:
            print('W_sn: \n', model.conv.weight.data.numpy()[0, 0, :, :])
        # compute and print gradients
        print('Gradients:')
        if spectral_normalization:
            getattr(model.conv, 'weight_orig').register_hook(lambda grad: print(grad.numpy()[0, 0, :, :]))
        else:
            getattr(model.conv, 'weight').register_hook(lambda grad: print(grad.numpy()[0, 0, :, :]))
        loss.backward()
        # perform update
        optimizer.step()
        # print updated weights
        if spectral_normalization:
            print('W_updated: \n', getattr(model.conv, 'weight_orig').detach().numpy()[0, 0, :, :])
        else:
            print('W_updated: \n', getattr(model.conv, 'weight').detach().numpy()[0, 0, :, :])


##############################
# tf forward + backward pass #
##############################

def tf_fwd(inp, label, n_iter, spectral_normalization=False):

    print('##########')
    print('TF Forward')
    print('##########')

    # create object with or without sn
    if spectral_normalization:
        conv = network_components.SpectralNormalization(
            tf.keras.layers.Conv2D(
                    1, kernel_size=(3, 3), strides=(1, 1), weights=[W_tf],
                    padding='same', input_shape=(3, 3, 1), use_bias=False))
        pool = tf.keras.layers.AveragePooling2D(pool_size=3)
    else:
        conv = tf.keras.layers.Conv2D(
                1, kernel_size=(3, 3), strides=(1, 1), weights=[W_tf],
                padding='same', input_shape=(3, 3, 1), use_bias=False)
        pool = tf.keras.layers.AveragePooling2D(pool_size=3)

    # create loss function and SGD optimizer
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    # initialize and print weights
    _ = conv(inp, training=False)
    if spectral_normalization:
        print('W_orig: \n', conv.layer.kernel_orig.numpy()[:, :, 0, 0])
    else:
        print('W_orig: \n', conv.kernel.numpy()[:, :, 0, 0])

    # perform forward and backward passes
    for i in range(n_iter):
        print('Forward Pass: ', i+1)

        with tf.GradientTape() as tape:
            # get and print output
            out11 = conv(inp, training=True)
            out1 = pool(out11)
            print('Output: ', out1.numpy())
            # compute loss
            loss = loss_fn(label, out1)
        # print spectrally normalized weights
        if spectral_normalization:
            print('W_sn: \n', conv.layer.kernel.numpy()[:, :, 0, 0])
        # compute and print gradients
        gradients = tape.gradient(loss, conv.trainable_variables)
        print('Gradients:\n', gradients[0].numpy()[:, :, 0, 0])
        # perform update and print weights
        optimizer.apply_gradients(zip(gradients, conv.trainable_variables))
        if spectral_normalization:
            print('W_updated: \n', conv.layer.kernel_orig.numpy()[:, :, 0, 0])
        else:
            print('W_updated: \n', conv.kernel.numpy()[:, :, 0, 0])


if __name__ == '__main__':

    # test spectral normalization
    sn = True

    # number of passes
    n_iter = 3

    # torch forward  and backward passes
    inp_t = torchmul.ones(1, 1, 3, 3)
    label_t = torchmul.ones(1, 1, 1, 1)

    torch_fwd(inp_t, label_t, n_iter, spectral_normalization=sn)

    # tf forward and backward passes
    inp_tf = tf.ones([1, 3, 3, 1])
    label_tf = tf.ones([1, 1, 1, 1])

    tf_fwd(inp_tf, label_tf, n_iter, spectral_normalization=sn)
