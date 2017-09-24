
""" This Python script converts the MobileNets weights to Metal CNN format.
    It uses the Caffe model from https://github.com/shicai/MobileNet-Caffe. """

import os
import sys
import numpy as np

caffemodel_file = "mobilenet.caffemodel"
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

            # This is a convolutional layer or the fc7 layer.
            if len(blob.shape.dim) == 4:
                c_o  = blob.shape.dim[0]
                c_i  = blob.shape.dim[1]
                h    = blob.shape.dim[2]
                w    = blob.shape.dim[3]
                print("  %d: %d x %d x %d x %d" % (idx, c_o, c_i, h, w))

                weights = np.array(blob.data, dtype=np.float32).reshape(c_o, c_i, h, w)
                layer_name = layer.name

            elif len(blob.shape.dim) == 2:
                weights = np.array(blob.data, dtype=np.float32)
                layer_name = layer.name

            elif len(blob.shape.dim) == 1:
                print("  %d: %d" % (idx, blob.shape.dim[0]))

                # This is a batch normalization layer.
                if layer.name[-3:] == "/bn":
                    if idx == 0:
                        mean = np.array(blob.data, dtype=np.float32)
                    elif idx == 1:
                        variance = np.array(blob.data, dtype=np.float32)

                # This is a scale layer. It always follows BatchNorm.
                elif layer.name[-6:] == "/scale":
                    if idx == 0:
                        gamma = np.array(blob.data, dtype=np.float32)
                    elif idx == 1:
                        if weights is None: print("*** ERROR! ***")
                        if mean is None: print("*** ERROR! ***")
                        if variance is None: print("*** ERROR! ***")
                        if gamma is None: print("*** ERROR! ***")

                        beta = np.array(blob.data, dtype=np.float32)

                        # We now have all the information we need to fold the batch 
                        # normalization parameters into the weights and bias of the
                        # convolution layer.

                        is_depthwise = layer_name[-3:] == "/dw"
                        if is_depthwise:
                            # In Caffe, the depthwise parameters are stored as shape
                            # (channels, 1, kH, kW). Convert to (kH, kW, channels).
                            
                            channel = weights.shape[0]
                            kh = weights.shape[2]
                            kw = weights.shape[3]
                            weights = weights.reshape(channel, kh, kw)
                            weights = weights.transpose(1, 2, 0)
                        else:
                            # Convert to (height, width, in_channels, out_channels).
                            # This order is needed by the folding calculation below.
                            weights = weights.transpose(2, 3, 1, 0)

                        conv_weights = weights * gamma / np.sqrt(variance + epsilon)

                        if not is_depthwise:
                            # Convert to (out_channels, height, width, in_channels),
                            # which is the format Metal expects.
                            conv_weights = conv_weights.transpose(3, 0, 1, 2)

                        conv_bias = beta - mean * gamma / np.sqrt(variance + epsilon)

                        out_name = layer_name + "_0.bin"
                        out_name = out_name.replace("/", "_")               
                        conv_weights.tofile(os.path.join(out_dir, out_name))
                        
                        out_name = layer_name + "_1.bin"
                        out_name = out_name.replace("/", "_")               
                        conv_bias.tofile(os.path.join(out_dir, out_name))
                    
                        weights = None
                        mean = None
                        variance = None
                        gamma = None
                        beta = None

                # This is the bias for the last layer (fc7)
                else:
                    if weights is None: print("*** ERROR! ***")

                    out_name = layer.name + "_0.bin"
                    out_name = out_name.replace("/", "_")               
                    weights.tofile(os.path.join(out_dir, out_name))

                    bias = np.array(blob.data, dtype=np.float32)
                    out_name = layer.name + "_1.bin"
                    out_name = out_name.replace("/", "_")               
                    bias.tofile(os.path.join(out_dir, out_name))

print("Done!")



