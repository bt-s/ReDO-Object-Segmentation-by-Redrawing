import torch
import torch.nn as nn
import torch_sn
import tensorflow as tf
import numpy as np
from collections import OrderedDict
import network_components

# torch formatted weights
np.random.seed(10)
W = np.random.randn(1, 1, 3, 3)
W_t = torch.from_numpy(W).float()

# tensorflow formatted weights
np.random.seed(10)
W = np.random.randn(3, 3, 1, 1)
W_tf = tf.convert_to_tensor(W)


# torch forward pass

def torch_fwd(inp, label, spectral_normalization=False):

    print('#############')
    print('Torch Forward')
    print('#############')

    torch.manual_seed(0)

    # create model with or without SN
    if spectral_normalization:
        conv_layer = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1),
                                padding=1, bias=False)
        conv_layer.weight.data = W_t
        model = torch.nn.Sequential(OrderedDict([
            ('conv', torch_sn.spectral_norm(conv_layer)),
            ('avg', torch.nn.AvgPool2d(3))
        ]))
    else:
        conv_layer = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1),
                                     padding=1, bias=False)
        conv_layer.weight.data = W_t
        model = torch.nn.Sequential(OrderedDict([
            ('conv', conv_layer),
            ('avg', torch.nn.AvgPool2d(3))
        ]))

    # create loss function and SGD optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    print('W_orig: \n', model.conv.weight.data.numpy()[0, 0, :, :])

    for i in range(1):
        print('Forward Pass ', i+1)

        model.zero_grad()
        out1 = model(inp)
        loss = loss_fn(out1, label)
        if spectral_normalization:
            print('W_sn: \n', model.conv.weight.data.numpy()[0, 0, :, :])
        print('Trainable Variables:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        print('Gradients:')
        getattr(model.conv, 'weight_orig').register_hook(lambda grad: print(grad.numpy()[0, 0, :, :]))
        loss.backward()
        optimizer.step()
        if spectral_normalization:
            print('W_updated: \n', getattr(model.conv, 'weight_orig').detach().numpy()[0, 0, :, :])
        else:
            print('W_updated: \n', getattr(model.conv, 'weight').detach().numpy()[0, 0, :, :])
        out2 = model(inp)
        print('Output before update: ', out1.detach().numpy())
        print('Output after update: ', out2.detach().numpy())

    return out1, out2


def tf_fwd(inp, label, spectral_normalization=False):

    print('##########')
    print('TF Forward')
    print('##########')

    if spectral_normalization:
        conv = network_components.SpectralNormalization(
            tf.keras.layers.Conv2D(
                    1, kernel_size=(3, 3), strides=(1, 1), weights=[W_tf], trainable=False,
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

    _ = conv(inp, training=False)
    if spectral_normalization:
        print('W_orig: \n', conv.layer.weights[0].numpy()[:, :, 0, 0])

    else:
        print('W_orig: \n', conv.weights[0].numpy()[:, :, 0, 0])

    for i in range(1):
        print('Forward Pass ', i+1)
        with tf.GradientTape() as tape:
            out11 = conv(inp, training=True)
            out1 = pool(out11)
            loss = loss_fn(label, out1)
        if spectral_normalization:
            print('W_sn: \n', conv.layer.weights[0].numpy()[:, :, 0, 0])
        print(conv.trainable_variables)
        gradients = tape.gradient(loss, conv.trainable_variables)
        print(gradients)
        print('Gradients:\n', gradients[0].numpy()[:, :, 0, 0])

        optimizer.apply_gradients(zip(gradients, conv.trainable_variables))
        print('Trainable:\n ', conv.trainable_variables[0].numpy()[:, :, 0, 0])
        if spectral_normalization:
            print('W_updated: \n', conv.layer.weights[0].numpy()[:, :, 0, 0])

        else:
            print('W_updated: \n', conv.weights[0].numpy()[:, :, 0, 0])
        out21 = conv(inp, training=True)
        out2 = pool(out21)
        print('Output before update: ', out1.numpy())
        print('Output after update: ', out2.numpy())
    return out1, out2


if __name__ == '__main__':

    # test spectral normalization
    sn = True

    # torch forward pass
    inp_t = torch.ones(1, 1, 3, 3)
    label_t = torch.ones(1, 1, 1, 1)

    out_bu, out_au = torch_fwd(inp_t, label_t, spectral_normalization=sn)

    # tf forward pass
    inp_tf = tf.ones([1, 3, 3, 1])
    label_tf = tf.ones([1, 1, 1, 1])

    out_bu, out_au = tf_fwd(inp_tf, label_tf, spectral_normalization=sn)
