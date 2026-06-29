# Utility that dumps bytes from a '.pt' file

# File format storage:
# -> simple binary file
# magic number     - 0x50 0x54
# 4 byte number    - matrix height (H)
# 4 byte number    - matrix width (W)
# H*W*4 numbers    - 32-bit floats in the matrix one by one in row-major format
# 1 byte separator - 0x00
#
# If a 2D matrix is followed by M>=1 N==1, it's interpreted as weights (the matrix) followed by a corresponding bias

import sys
import struct
from typing import BinaryIO


MAGIC = '\x50\x54'.encode('utf-8')
SEPARATOR = '\x00'.encode('utf-8')

WEIGHT_LAYER_NAME = "weight"
BIAS_LAYER_NAME = "bias"

def dump_shape(writer: BinaryIO, height: int, width: int):
    writer.write((height).to_bytes(4))
    writer.write((width).to_bytes(4))

def dump_vector(writer: BinaryIO, data: torch.Tensor):
    if len(data.shape) != 1:
        raise Exception("dump_vector function needs 1D tensor")
    packed_data = struct.pack('f' * len(data), *list(data))
    writer.write(packed_data)

def dump_matrix(writer: BinaryIO, data: torch.Tensor):
    if len(data.shape) != 2:
        raise Exception("dump_weight function needs 2D tensor")
    for row in data:
        dump_vector(writer, row)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"{sys.argv[0]} needs 2 arguments: <pytorch .pt file> <output bytes file>")
        exit(1)

    # Import here to prevent loading entire pytorch library if arguments incorrect
    # (since this is a simple utility file and loading pytorch may take a bit)
    import torch

    infile = sys.argv[1]
    outfile = sys.argv[2]

    model = torch.load(infile)

    with open(outfile, 'wb') as fw:
        fw.write(MAGIC)
        for key in model.keys():
            if WEIGHT_LAYER_NAME in key:
                shape = model[key].shape
                dump_shape(fw, shape[0], shape[1])
                dump_matrix(fw, model[key])
            elif BIAS_LAYER_NAME in key:
                dump_shape(fw, len(model[key]), 1)
                dump_vector(fw, model[key])
            fw.write(SEPARATOR)

