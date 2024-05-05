# Value

This folder contains modules to build a computation graph using Value objects. The Value is an atomic component of building a neural network. We can use the Value to build Neuron, Layers of Neurons, and a full NeuralNetwork model.

We illustrate how to build a classification model on the make_moon dataset.

1. Create an environment: `python -m venv venv`
2. Activate the environment: `source venv/bin/activate` (to deactivate just run `deactivate`)
3. Install requirements: `pip install -r requirements.txt`
4. To run all tests: `python -m unittest -v`
5. To train a model: `python -m train`
