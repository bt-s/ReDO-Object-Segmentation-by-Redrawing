import torch

import tensorflow as tf

import numpy as np
from collections import OrderedDict

import network_components


# torch formatted weights
np.random.seed(10)
W = np.random.randn(1, 1, 3, 3)
#W = np.reshape(W, (1, 1, 3, 3))
W_t = torch.from_numpy(W).float()

# tensorflow formatted weights
np.random.seed(10)
W = np.random.randn(3, 3, 1, 1)
W_tf = tf.convert_to_tensor(W)

def torch_fwd(inp, label):
    torch.manual_seed(0)
    model = torch.nn.Sequential(OrderedDict([
        ('conv', torch.nn.utils.spectral_norm(
            torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1),
                            padding=1, bias=False))
        ),
        ('avg', torch.nn.AvgPool2d(3))
    ]))
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.conv.weight.data = W_t
    out1 = model(inp)

    for _ in range(1):
        model.zero_grad()
        out1 = model(inp)
        loss = loss_fn(out1, label)
        loss.backward()
        optimizer.step()

    return out1, model(inp)


def tf_fwd(inp, label):
    model = tf.keras.Sequential()
    model.add(
        network_components.SpectralNormalization(
            tf.keras.layers.Conv2D(
                1, kernel_size=(3, 3), strides=(1, 1), weights=[W_tf],
                padding='same', input_shape=(3, 3, 1), use_bias=False
            )
        )
    )
    model.add(
        tf.keras.layers.AveragePooling2D(pool_size=3)
    )

    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    out1 = model(inp)

    for _ in range(1):
        with tf.GradientTape() as tape:
            out1 = model(inp)
            loss = loss_fn(label, out1)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.layers[0].reassign_orig()
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return out1, model(inp)

inp_t = torch.ones(1, 1, 3, 3)
label_t = torch.ones(1, 1, 1, 1)

print(torch_fwd(inp_t, label_t))

inp_tf = tf.ones([1, 3, 3, 1])
label_tf = tf.ones([1, 1, 1, 1])

print(tf_fwd(inp_tf, label_tf))