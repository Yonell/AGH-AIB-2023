from enum import Enum
import numpy as np
import scipy as sp


def identity_fn(x):
    return x


def sigmoid_fn(x):
    return sp.special.expit(x)


def tanh_fn(x):
    return np.tanh(x)


def relu_fn(x):
    return np.maximum(0, x)


def leaky_relu_fn(x):
    return np.maximum(0.01 * x, x)


def softmax_fn(x):
    return np.exp(x) / np.sum(np.exp(x))


class ActivationFunction(Enum):
    def __call__(self, x):
        return self.value(x)

    def __str__(self):
        return self.value

    identity = identity_fn
    sigmoid = sigmoid_fn
    tanh = tanh_fn
    relu = relu_fn
    leaky_relu = leaky_relu_fn
    softmax = softmax_fn


availible_activation_functions = [ActivationFunction.identity, ActivationFunction.sigmoid, ActivationFunction.tanh,
                                  ActivationFunction.relu, ActivationFunction.leaky_relu, ActivationFunction.softmax]
