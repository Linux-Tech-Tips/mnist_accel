# File running simple numpy inference with a given data file of tensors and png image path

from __future__ import annotations

import sys
import struct
import time
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

import image_loader


# basic ReLU activation function for numpy arrays
def relu(x: np.array) -> np.array:
    return (abs(x) + x) / 2

@dataclass
class NetworkLayer:
    """Single network layer represented as a 2D NumPy Matrix for the weights, 1D NumPy array for bias and an activation function"""
    weights: np.array
    bias: np.array
    activation: Callable[[np.array], np.array]

    def __post_init__(self):
        """Standard constraints for weights and bias in any layer"""
        if len(self.weights.shape) != 2:
            raise Exception("NetworkLayer: weights must be a 2D array/matrix")
        if len(self.bias.shape) != 1:
            raise Exception("NetworkLayer: bias must be 1D array")
        if self.bias.shape[0] != self.weights.shape[0]:
            raise Exception("NetworkLayer: bias must be same length as weights height")

    def run(self, in_layer: np.array) -> np.array:
        return self.activation((np.matmul(self.weights, in_layer) + self.bias))

@dataclass
class Network:
    """Network layers as NumPy arrays"""
    layers: list[NetworkLayer]

    def __post_init__(self):
        """Standard constraints for loaded network"""
        for layer_idx in range(1, len(self.layers)):
            if self.layers[layer_idx].weights.shape[1] != self.layers[layer_idx - 1].weights.shape[0]:
                raise Exception("Network: layer in/out dimensions mismatch")

    def run(self, in_layer: np.array) -> int:
        """Runs network with given input layer 1D array"""
        if len(in_layer.shape) != 1:
            raise Exception("Network: needs 1D array input")
        tmp_array = None
        for layer in self.layers:
            tmp_array = layer.run(in_layer if tmp_array is None else tmp_array)
        return np.argmax(tmp_array).item()

    @staticmethod
    def load_from_dat_file(path: str) -> Network:
        layers = []
        with open(path, "rb") as fr:
            magic = fr.read(2).decode("utf-8")
            if magic != "PT":
                raise Exception("Network: load from file needs custom data format")
            layer_num = 0
            tmp_layer = None
            while True:
                buff = fr.read(4)
                if len(buff) < 4:
                    if len(buff) != 0:
                        raise Exception("Network: file load error, mismatched bytes")
                    # EOF reached
                    break
                dim_h = int.from_bytes(buff)
                dim_w = int.from_bytes(fr.read(4))
                # Read weights
                rows = []
                for idx in range(dim_h):
                    rows.append(list(struct.unpack('f' * dim_w, fr.read(dim_w * 4))))
                weights = np.stack(rows)
                if fr.read(1) != b'\x00':
                    raise Exception(f"Network: expected zero byte separator after weights in layer {layer_num}")
                # Read bias
                bias_len = int.from_bytes(fr.read(4))
                if int.from_bytes(fr.read(4)) != 1:
                    raise Exception(f"Network: expected bias W dimension of 1 in layer {layer_num}")
                bias = np.array(list(struct.unpack('f' * bias_len, fr.read(bias_len * 4))))
                layers.append(NetworkLayer(weights, bias, relu))
                if fr.read(1) != b'\x00':
                    raise Exception(f"Network: expected zero byte separator after bias in layer {layer_num}")
                layer_num += 1
        # Create network instance from layers
        return Network(layers)

# --- convenience functions ---

def run_net_on_image(net: Network, path: str) -> int:
    image = image_loader.png_to_numpy(path)
    return net.run(image.flatten())
    
def run_perf_test(network_path: str, image_paths: list, repeat: int) -> tuple[float, float]:
    """Runs 'repeat' tests executing the network given by network_path on all images in image_path sequentially
       Returns a tuple of (<model load time>, <total test time>) both floating point in seconds"""

    # Model loading
    model_load_start = time.time()
    net = Network.load_from_dat_file(network_path)
    model_load_end = time.time()

    # Image loading
    images = [image_loader.png_to_numpy(path).flatten() for path in image_paths]

    # Model inference benchmark
    run_start = time.time()
    for i in range(repeat):
        for image in images:
            net.run(image)
    run_end = time.time()

    return (model_load_end - model_load_start, run_end - run_start)


# Runtime
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"{sys.argv[0]} needs first argument 'run' or 'bench'")
        print(f" - run:   {sys.argv[0]} run network_file.dat image_file.png")
        print(f" - bench: {sys.argv[0]} bench network_file.dat repeats image_files.png ...")
        raise SystemExit(1)

    if sys.argv[1].lower() == "run":
        if len(sys.argv) < 4:
            print("'run' needs 2 additional arguments")
            raise SystemExit(1)
        net = Network.load_from_dat_file(sys.argv[2])
        result = run_net_on_image(net, sys.argv[3])
        print(f"Result: {result}")

    elif sys.argv[1].lower() == "bench":
        if len(sys.argv) < 5:
            print("'bench' needs at least 3 additional arguments")
            raise SystemExit(1)
        model_path = sys.argv[2]
        repeats = int(sys.argv[3])
        image_paths = [sys.argv[i] for i in range(4, len(sys.argv))]
        load_time, run_time = run_perf_test(model_path, image_paths, repeats)
        load_time *= 1000
        run_time *= 1000
        run_time_per_img = run_time / (len(image_paths) * repeats)
        print(f"Result:\nTest repeats: {repeats}; Image count: {len(image_paths)}; Model load time: {load_time} ms \
                \nInference time total: {run_time} ms; Inference time average: {run_time_per_img} ms")

