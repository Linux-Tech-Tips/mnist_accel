# MLP FNN Training Segment

This part of the project consists of PyTorch Python code to train a multi-layer perceptron fully connected network on the MNIST dataset.

## Model Parameters

### Layers:

- **Input**: 784 neurons, flattened 28x28 pixel image
- **Hidden 1**: 512 neurons, ReLU activation
- **Hidden 2**: 256 neurons, ReLU activation
- **Output**: 10 neurons representing digit 0-9, ArgMax on inference for simplicity

The ReLU function is used as it both avoids the vanishing gradient problem,
and will be much simpler and faster to implement in the second stage of this project.

ReLU also has a good track record as a general-purpose activation function, so it's sufficient for this project.

### Optimiser:

- ADAM Optimiser thanks to good performance, with a configurable learning rate of $10^-4$ by default.

## Data Dump Script

For ease of use, along with the `main.py` script allowing model training, saving, loading and inference, the script `dump_bytes.py` is provided.

The `dump_bytes.py` script takes a PyTorch (.pt) model saved by the `main.py` script, and dumps it into a byte file with a custom, simplified format
for ease of loading in the inference component of this project.

The format, also documented in a comment in the `dump_bytes.py` script itself, is as follows:

- 2 bytes ... magic numbers 0x50 0x54 (ASCII for 'P' and 'T', meaning the file contains PyTorch Tensors)
- stream of X datagrams, following format:
  - 4 bytes ....... integer tensor dimension H (typically tensor height)
  - 4 bytes ....... integer tensor dimension W (typically tensor width)
  - H*W*4 bytes ... series of 32-bit floating point numbers (IEEE 754), tensor data in row-major format
  - 1 byte ........ separator 0x00 signalling end of the datagram, sanity check that previous data loaded correctly

Given a model with layers containing the name 'weight' and 'bias', the script saves the weights and biases in alternating order,
as they appear layer by layer, with weight matrices saved in usual HxW row-major format, and biases saved in a single line with the shape H>=1, W==1.

