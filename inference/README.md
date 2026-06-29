# MLP FNN Inference Segment

This part of the project consists of NumPy Python code, basic C code and Arm64-optimised C code capable of pre-trained multi-layer perceptron model inference.

## Example Data

In the directory `./data/` are sample data provided for convenience, notably `default.pt`, a PyTorch model trained to approx. 98.4% accuracy,
and 10 sample PNG images of digits 0-9 that can be used to run the model.

The `./data/` directory also contains a simplified file `default.dat`, which is a custom export of the PyTorch tensors in a simplified data file.
The file is created using the `../training/dump_bytes.py` script, and the format is documented in the `../training/` directory.
For convenience, both the PyTorch export file and the simplified data dump file (used for ease of loading here) are provided in the `./data/` directory.

## NumPy Code

TODO Add
