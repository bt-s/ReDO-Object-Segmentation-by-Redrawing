#!/usr/bin/python3

"""test_spectral_norm.py - Script to test whether our TensorFlow 2.0
                           implementation of spectral normalization is
                           correct.
For the NeurIPS Reproducibility Challenge and the DD2412 Deep Learning, Advanced
course at KTH Royal Institute of Technology.
"""

__author__ = "Adrian Chiemelewski-Anders, Mats Steinweg & Bas Straathof"


from typing import Tuple, List

import unittest
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import torch

import network_components

USE_SPECTRAL_NORM = True

def configure_harness():
    tf.keras.backend.set_floatx('float64')


def permute_shape(shape: Tuple, order: Tuple) -> Tuple:
    """Take a shape and permute it. i.e. change the order of the elements

    Args:
        shape: Input shape
        order: Ordering of shape (i.e. [0, len(shape)))

    Returns:
        A tuple with the same values as `shape` but with the order
        corresponding to `order`
    """
    permuted_shape = []
    for i in order:
        permuted_shape.append(shape[i])

    return tuple(permuted_shape)


def create_random_tensors_nd(shape_t: Tuple, order: Tuple) -> Tuple[
    torch.Tensor, tf.Tensor]:
    """Create a random torch tensor, and a tensorflow tensor with same
    dimension and values but different ordering corresponding to `order`

    Args:
        shape_t: Shape that the torch tensor will take on
        order: Permutation of `shape_t` which the tensorflow tensor will
               assume
    Returns:
        a tuple, one torch tensor and one tensorflow tensor with the
        aforementioned specifications
    """
    sigma = np.sqrt(2.0 / shape_t[1])
    tensor_t = np.random.normal(0, sigma, shape_t)
    tensor_tf = np.transpose(np.copy(tensor_t), order)

    return torch.from_numpy(tensor_t), tf.convert_to_tensor(tensor_tf)


def copy_t_tensor_to_tf(tensor_t: torch.Tensor, order: Tuple) -> tf.Tensor:
    """Duplicate a torch tensor to a tensorflow tensor with the specified
    reordering

    Args:
        tensor_t: Torch tensor to duplicate
        order: Specifies the permutation

    Returns:
        TensorFlow tensor with duplicate elements but with indices permuted
    """
    tensor_np = np.copy(np.array(tensor_t.tolist()))
    tensor_np = np.transpose(tensor_np, order)

    return tf.convert_to_tensor(tensor_np)


def convert_t_model_to_tf_weights(model_t: torch.nn.Module) -> List[tf.Tensor]:
    """Convert a torch model to a list of tensorflow weights

    Args:
        model_t: The torch model to convert
    Returns:
        W_tf: Corresponding to the proper weights for a tensorflow model
    """
    orders = [None, None, (1, 0), None, (2, 3, 1, 0)]
    W_tf = []
    for _, weight in model_t.named_parameters():
        weights_to_copy = weight.data

        W_tf.append(copy_t_tensor_to_tf(weights_to_copy,
            orders[len(weights_to_copy.shape)]))

    return W_tf


def backward_t(model: torch.nn.Module, inp: torch.Tensor,
        label: torch.Tensor) -> torch.nn.Module:
    """Perform one backward step on `model` with `inp` and `label`

    Args:
        model: The torch model
        inp: The input
        label: The label for `inp`

    Returns:
        model: The same model, but with the weights updated
    """
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.zero_grad()
    out = model(inp)
    loss = loss_fn(out, label)
    loss.backward()
    optimizer.step()

    return model


def backward_tf(model: tf.keras.Model, inp: tf.Tensor,
        label: tf.Tensor) -> tf.keras.Model:
    """Perform one backward pass of `model` using `inp` and `label`

    Args:
        model: The tensorflow/keras model
        inp: The input
        label: The label for `inp`

    Returns:
        model: The same model, but with the weights updated
    """
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    with tf.GradientTape() as tape:
        out = model(inp)
        loss = loss_fn(label, out)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return model


def forward_outputs(model_t: torch.nn.Module, model_tf: tf.keras.Model,
        shape_t: Tuple, order: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a forward pass on a torch and tensorflow model, and return
    their outputs in numpy format. Inputs are created randomly

    Args:
        model_t: A torch model
        model_tf: A tensorflow model
        shape_t: The shape of the input for the torch model
        order: A permutation of `shape_t` see `permute_shape` and
               `create_random_tensors_nd`
    Returns:
        a tuple of np.ndarrays, the shape is not altered from what the models
        produce
    """
    inp_t, inp_tf = create_random_tensors_nd(shape_t, order)

    return np.array(model_t(inp_t).tolist()), model_tf(inp_tf).numpy()


def backward_forward_outputs(input_shape_t: Tuple, order: Tuple,
        model_t: torch.nn.Module, model_tf: tf.keras.Model) -> Tuple[np.ndarray,
                np.ndarray]:
    """Perform a backward pass for both `model_t` and `model_tf` and then a
    forward pass and return the outputs

    Args:
        input_shape_t: Input shape for the torch model
        order: Permutation of `input_shape_t` to the input shape of the
               tensorflow model
        model_t: The torch model
        model_tf: The tensorflow model

    Returns:
        See `forward_outputs`
    """
    inp_t, inp_tf = create_random_tensors_nd(input_shape_t, order)
    batches = input_shape_t[0]
    label_t, label_tf = create_random_tensors_nd((batches, 1, 1, 1),
            (0, 1, 2, 3))

    model_t = backward_t(model_t, inp_t, label_t)
    model_tf = backward_tf(model_tf, inp_tf, label_tf)

    return forward_outputs(model_t, model_tf, input_shape_t, order)


class TestSpectralNormSmall(unittest.TestCase):
    def _build_torch_model(self, W: torch.Tensor) -> torch.nn.Module:
        """Create the torch model for the test

        Args:
            W: Weights for the single conv layer
        Returns:
            model: the torch model
        """
        conv_layer = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1),
                        padding=1, bias=False)
        conv_layer.weight.data = W
        model = torch.nn.Sequential(torch.nn.utils.spectral_norm(conv_layer),
                torch.nn.AvgPool2d(3))

        return model


    def _build_tf_model(self, W: tf.Tensor) -> tf.keras.Model:
        """Create the same model as `_build_torch_model` but in tensorflow

        Args:
            W: Weights for the single conv layer
        Returns:
            model: The created model
        """
        conv = network_components.SpectralNormalization(
            tf.keras.layers.Conv2D(1, kernel_size=(3, 3), strides=(1, 1),
                 weights=[W], padding='same',input_shape=self.input_shape_tf,
                 use_bias=False))

        pool = tf.keras.layers.AveragePooling2D(pool_size=3)

        model = tf.keras.Sequential()
        model.add(conv)
        model.add(pool)

        return model


    def setUp(self):
        """Set up the test class for each test"""
        self.batches = 5
        W_t, W_tf = create_random_tensors_nd((1, 1, 3, 3), (2, 3, 0, 1))

        # For inputs torch orders like: (batch, channels, width, height)
        # and TensorFlow orders like  : (batch, width, height, channels)
        self.input_shape_t = (self.batches, 1, 3, 3)
        self.order = (0, 2, 3, 1)
        self.input_shape_tf = permute_shape(self.input_shape_t, self.order)

        self.model_t = self._build_torch_model(W_t)
        self.model_tf = self._build_tf_model(W_tf)


    def test_forward(self):
        """Test that one forward pass is the same for both models"""
        out_t, out_tf = forward_outputs(
            self.model_t, self.model_tf,
            self.input_shape_t, self.order)

        self.assertEqual(out_t.shape, out_tf.shape)
        np.testing.assert_array_almost_equal(out_t, out_tf, decimal=5)


    def test_backward(self):
        """Test that one backward pass is the same for both models"""
        out_t, out_tf = backward_forward_outputs(self.input_shape_t, self.order,
                self.model_t, self.model_tf)
        self.assertEqual(out_t.shape, out_tf.shape)
        np.testing.assert_array_almost_equal(out_t, out_tf, decimal=5)


class TorchModelLarge(torch.nn.Module):
    def __init__(self, channels: int):
        """Create the model

        Args:
            channels: how many channels in the input
        """
        super(TorchModelLarge, self).__init__()

        self.conv1 = torch.nn.Conv2d(channels, 8, kernel_size=(3, 3),
                                     bias=False).double()
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=(3, 3), bias=False).double()
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flat = torch.nn.Flatten()
        self.lin1 = torch.nn.Linear(144, 32, bias=False).double()
        self.lin2 = torch.nn.Linear(32, 1, bias=False).double()

        if USE_SPECTRAL_NORM:
            self.conv1 = torch.nn.utils.spectral_norm(self.conv1).double()
            self.conv2 = torch.nn.utils.spectral_norm(self.conv2).double()


    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass
        Args:
            inp: The input
        Return:
            out: The result of a forward pass
        """
        out = self.conv1(inp)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = out.permute(0, 2, 3, 1)
        out = self.flat(out)
        out = self.lin1(out)
        out = self.lin2(out)
        out = self.relu(out)

        return out


class TestSpectralNormLarge(unittest.TestCase):
    class NullLayer(tf.keras.layers.Layer):
        def __init__(self):
            super(TestSpectralNormLarge.NullLayer, self).__init__()

    def _build_tf_model(self, W: List[tf.Tensor]) -> tf.keras.Model:
        """Build the tensorflow model from the torch model weights

        Args:
            W: Weights recovered and duplicated from the torch model

        Returns:
            model: The built model with proper weights
        """
        model = tf.keras.Sequential()

        conv1 = tf.keras.layers.Conv2D(8, 3, activation='relu', use_bias=False,
                input_shape=self.input_shape_tf[1:], weights=[W[0]])
        conv2 = tf.keras.layers.Conv2D(16, 3, activation='relu', use_bias=False,
                weights=[W[1]])

        if USE_SPECTRAL_NORM:
            conv1 = network_components.SpectralNormalization(conv1)
            conv2 = network_components.SpectralNormalization(conv2)

        model.add(conv1)
        model.add(conv2)

        model.add(tf.keras.layers.MaxPool2D(strides=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(32, use_bias=False, weights=[W[2]]))
        model.add(tf.keras.layers.Dense(1, use_bias=False, activation='relu',
            weights=[W[3]]))

        return model


    def setUp(self):
        """Configure models before each test"""
        self.batches = 2
        self.channels = 3
        self.im_size = 10
        self.input_shape_t = (self.batches, self.channels, self.im_size,
                self.im_size)
        self.order = (0, 2, 3, 1)

        self.input_shape_tf = permute_shape(self.input_shape_t, self.order)

        self.model_t = TorchModelLarge(self.channels)
        self.model_tf = self._build_tf_model(convert_t_model_to_tf_weights(
            self.model_t))


    def test_forward(self):
        """Test that the forward passes produce the same output"""
        out_t, out_tf = forward_outputs(self.model_t, self.model_tf,
                self.input_shape_t, self.order)

        print(out_t)
        print('fwd---')
        print(out_tf)
        self.assertEqual(out_t.shape, out_tf.shape)
        np.testing.assert_array_almost_equal(out_t, out_tf, decimal=5)


    def test_backward(self):
        """Test that the backward passes update the weights, producing the
        same forward outputs for both models"""
        out_t, out_tf = backward_forward_outputs(self.input_shape_t, self.order,
                self.model_t, self.model_tf)

        print(out_t)
        print('bck---')
        print(out_tf)
        self.assertEqual(out_t.shape, out_tf.shape)
        np.testing.assert_array_almost_equal(out_t, out_tf, decimal=5)

if __name__ == '__main__':
    configure_harness()
    unittest.main()

