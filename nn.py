import numpy as np
from value import Value

class Neuron():
    def __init__(self, input_size, activation):
        self.w = np.array([Value(np.random.normal()) for _ in range(input_size+1)])
        self.activation = activation

    def __call__(self, input):
        out = np.dot(input, self.w[:-1]) + self.w[-1]
        if self.activation is not None:
            out = self.activation(out)
        return out

    def zero_grad(self):
        for i in range(len(self.w)):
            self.w[i].grad = 0.0

class Layer():
    def __init__(self, input_size, output_size, activation):
        self.neurons = np.array([Neuron(input_size, activation) for _ in range(output_size)])

    def __call__(self, input):
        return np.array([neuron(input) for neuron in self.neurons])

    def zero_grad(self):
        for neuron in self.neurons:
            neuron.zero_grad()
        
class NeuralNetwork():
    # We expect the sizes of output_sizes, activations be equal.
    def __init__(self, input_size, output_sizes, activations):
        layers = []
        for output_size, activation in zip(output_sizes, activations):
            layers.append(Layer(input_size, output_size, activation))
            input_size = output_size
        self.layers = np.array(layers)

    def params(self):
        out = []
        for layer in self.layers:
            for neuron in layer.neurons:
                for p in neuron.w:
                    out.append(p)
        return np.array(out)

    def _forward(self, input):
        out = input
        for layer in self.layers:
            out = layer(out)
        return out

    def forward(self, inputs):
        scores = []
        for input in inputs:
            scores.append(self._forward(input))
        return np.array(scores)
    
    def step(self, learning_rate):
        for layer in self.layers:
            for neuron in layer.neurons:
                for i in range(len(neuron.w)):
                    neuron.w[i].data -= learning_rate * neuron.w[i].grad

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def train(self, inputs, labels, epochs=100, learning_rate=0.01, batch_size=None, regularization_param=1e-4):
        losses = []
        for i in range(epochs):
            loss = Value(0)
            if batch_size is not None:
                indices = np.random.permutation(inputs.shape[0])[:batch_size]
                inputs, labels = inputs[indices], labels[indices]
            scores = self.forward(inputs)
            for p, y in zip(scores, labels):
                loss -= y[0] * p[0].log() + (1-y[0]) * (1-p[0]).log()
            loss /= len(scores)
            if regularization_param > 0.0:
                loss += regularization_param * sum([p*p for p in self.params()])
            losses.append(loss.data)
            self.zero_grad()
            loss.backward()
            self.step(learning_rate)
            if i % 1 == 0:
                print(f'step: {i+1}, loss: {loss.data}')
        return losses, scores, labels
