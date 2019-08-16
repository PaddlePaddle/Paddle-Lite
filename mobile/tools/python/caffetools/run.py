import caffe
import numpy as np

prototxt_path = ""
caffemodel_path = ""
input_path = "input.txt"
input_name = ""
output_name = ""

shape = (1, 3, 64, 64)

data = np.loadtxt(input_path).astype("float32").reshape(shape)

net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

# view inputs blob names
print(net.inputs)

# view outputs blob names
print(net.outputs)

# set input data
net.blobs[input_name].reshape(*shape)
net.blobs[input_name].data[...] = data

# predict
net.forward()

# view output data
print(net.blobs[output_name].data)
