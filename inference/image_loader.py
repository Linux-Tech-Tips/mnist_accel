# Image loader using python to load png images for MNIST inference
# Has capability to be used both to load data from another Python script
# or to serve as a standalone converter to a binary format much easier to load for the C inference

import sys
import struct

import numpy as np
from PIL import Image

# Unless specified otherwise, for convenience loads as float32
def png_to_numpy(filepath: str, load_float: bool = True) -> np.array:
    array = np.array(Image.open(filepath))
    if load_float:
        array = array.astype(np.float32) / 255
    return array

# image datafile magic 'IM' for image
_DATAFILE_MAGIC = '\x49\x4D'.encode('utf-8')

def _save_png_to_bytes(png_path: str, data_path: str):
    array = png_to_numpy(png_path)
    with open(data_path, 'wb') as fw:
        fw.write(_DATAFILE_MAGIC)
        for row in array:
            packed_row = struct.pack('f' * len(row), *list(row))
            fw.write(packed_row)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"{sys.argv[0]} needs 2 arguments: <png image file path> <output data file path>")
        exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]

    _save_png_to_bytes(infile, outfile)
