import numpy as np

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

import nn
import activation as act
from value import Value

def Accuracy(scores, labels, classification_threshold=0.5):
    return np.mean([(score[0].data > classification_threshold) == (label[0].data > classification_threshold) for score, label in zip(scores, labels)])


if __name__ == '__main__':
    model = nn.NeuralNetwork(2, [16, 16, 1], [act.tanh, act.tanh, act.sigmoid])
    
    X, y = make_moons(noise=0.2, n_samples=100, shuffle=True)
    y = y.reshape(-1, 1)
    
    VecValue = np.vectorize(Value)
    X, y = VecValue(X), VecValue(y)
    
    losses, scores, labels = model.train(X, y, epochs=100, learning_rate=0.3, batch_size=10)
        
    print(f'final loss: {losses[-1]}')
    acc = Accuracy(scores, labels)
    print(f'Accuracy: {100*acc:3.0f}%')
    
    plt.plot(losses);
    plt.show()
    