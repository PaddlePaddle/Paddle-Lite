# -*- coding: utf-8 -*-
"""convert caffe model to the format the metal can read"""

import os
import sys
import numpy as np

caffemodel_file = "squeezenet_v1.1.caffemodel"
out_dir = "./Parameters"

print("Loading the Caffe model...")
import caffe_pb2
data = caffe_pb2.NetParameter()
data.MergeFromString(open(caffemodel_file, "rb").read())
layers = data.layer

layer_name = None
weights = None
mean = None
variance = None
gamma = None
epsilon = 1e-5

for layer in layers:
    if layer.blobs:
        print(layer.name)

        for idx, blob in enumerate(layer.blobs):

            if len(blob.shape.dim) == 4:
                c_o  = blob.shape.dim[0]
                c_i  = blob.shape.dim[1]
                h    = blob.shape.dim[2]
                w    = blob.shape.dim[3]
                print("  %d: %d x %d x %d x %d" % (idx, c_o, c_i, h, w))

                weights = np.array(blob.data, dtype=np.float32).reshape(c_o, c_i, h, w)
                weights = weights.transpose(0, 2, 3, 1)
                layer_name = layer.name

                out_name = layer.name + "_0.bin"
                out_name = out_name.replace("/", "_")               
                weights.tofile(os.path.join(out_dir, out_name))

            elif len(blob.shape.dim) == 1:   
                bias = np.array(blob.data, dtype=np.float32)
                out_name = layer.name + "_1.bin"
                out_name = out_name.replace("/", "_")               
                bias.tofile(os.path.join(out_dir, out_name))

print("Done!")
