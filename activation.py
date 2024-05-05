import numpy as np
from value import Value

def relu(value):
    out = Value(value.data if value.data > 0 else 0, 'relu', (value,))

    def backward():
        value.grad += (value.data > 0) * out.grad

    out.backward_ = backward
    return out

def tanh(value):
    y = np.tanh(value.data)
    out = Value(y, 'tanh', (value,))

    def backward():
        value.grad += (1.0 - y*y) * out.grad

    out.backward_ = backward
    return out

def sigmoid(value):
    y = 1 / (1 + np.exp(-value.data))
    out = Value(y, 'sigmoid', (value,))

    def backward():
        value.grad += y*(1 - y) * out.grad

    out.backward_ = backward
    return out