# MLP FNN Training Segment

This part of the project consists of PyTorch Python code to train a multi-layer perceptron fully connected network on the MNIST dataset.

## Model Parameters

### Layers:

- **Input**: 784 neurons, flattened 28x28 pixel image
- **Hidden 1**: 256 neurons, ReLU activation
- **Hidden 2**: 128 neurons, ReLU activation
- **Output**: 10 neurons representing digit 0-9, ArgMax on inference for simplicity

The ReLU function is used as it both avoids the vanishing gradient problem,
and will be much simpler and faster to implement in the second stage of this project.

ReLU also has a good track record as a general-purpose activation function, so it's sufficient for this project.

### Optimiser:

- Stochastic Gradient Descent (SGD) for simplicity

