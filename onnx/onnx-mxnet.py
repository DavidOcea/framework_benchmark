import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import numpy as np

# Import the ONNX model into MXNet's symbolic interface
sym, arg, aux = onnx_mxnet.import_model("../output/torch_model.onnx")
print("Loaded torch_model.onnx!")
print(sym.get_internals())