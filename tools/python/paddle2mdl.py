"""
Converting PaddlePaddle model to mdl format
"""
import os
import struct
import numpy as np
import caffe

"""
Converting model conf
"""


class ModelConfig(object):
    """
    init config
    """

    def __init__(self, path, params):
        self.model_path = path
        self.model_params = params


def load_model_parameters(conf, layers, output_path):
    """
    load model parameters
    """
    net = caffe.Net(conf.model_path, 1)
    if len(layers) == 0: layers = net.params.keys()
    param_num = 0
    p1 = np.asarray([0.406, 0.456, 0.485], dtype=np.float32)
    p2 = np.asarray([0.225 * 0.225, 0.224 * 0.224, 0.229 * 0.229], dtype=np.float32)
    # p2 *= 255.0 *255.0
    p = [p1, p2, np.asarray([1])]
    for layer_name in layers:
        params = net.params[layer_name]
        layer_name_new = layer_name.replace('/', '-')
        if "scale" in layer_name_new:
            print "rename"
            layer_name_new = layer_name_new.replace("scale", "bn")
        print layer_name_new

        param_num += len(params)

        for i in range(len(params)):
            if "bn" in layer_name:
                file = os.path.join(output_path, '_%s.w%s' % (layer_name_new, str(i + 1)))
            elif "scale" in layer_name or "fc" in layer_name or 'output' in layer_name:
                suffix = "0" if i == 0 else "bias"
                file = os.path.join(output_path, '_%s.w%s' % (layer_name_new, suffix))
            else:
                if i == 0:
                    file = os.path.join(output_path, '_%s.w%s' % (layer_name_new, str(i)))
                else:
                    file = os.path.join(output_path, '_%s.w%s' % (layer_name_new, 'bias'))
            print "loading for layer %s." % file
            if 'bn.w3' in file:
                data = np.asarray([1])
            elif layer_name == 'data_bn':
                data = p[i]
            else:
                data = load_parameter(file)
            print data.shape
            print params[i].shape
            data = np.reshape(data, params[i].shape)
            reference = np.array(params[i].data)
            if "fc" in layer_name or 'output' in layer_name:
                data = np.reshape(data, reference.T.shape)
                data = np.transpose(data)
            params[i].data[...] = data
    print "param_num = %d" % param_num
    net.save(conf.model_params)


def save_model_parameters(conf, layers, output_path):
    """
    save model parameters
    """
    net = caffe.Classifier(conf.model_path, conf.model_params,
                           image_dims=(256, 256),
                           channel_swap=(2, 1, 0),
                           raw_scale=255)
    if len(layers) == 0: layers = net.params.keys()
    param_num = 0
    for layer_name in layers:
        params = net.params[layer_name]
        layer_name_new = layer_name.replace('/', '-')

        if "scale" in layer_name_new:
            layer_name_new.replace("scale", "bn")

        param_num += len(params)
        for i in range(len(params)):
            data = np.array(params[i].data)
            if "bn" in layer_name:
                file = os.path.join(output_path, '_%s.w%s' % (layer_name_new, str(i + 1)))
            elif "scale" in layer_name or "fc" in layer_name or "output" in layer_name:
                suffix = "0" if i == 0 else "bias"
                file = os.path.join(output_path, '_%s.w%s' % (layer_name_new, suffix))
            else:
                file = os.path.join(output_path, '_%s.w%s' % (layer_name_new, str(i)))
            print "Saving for layer %s." % layer_name
            print "Saving for layer %s." % file
            print data.shape, data.size
            if "fc" in layer_name or "output" in layer_name:
                data = np.transpose(data)
            write_parameter(file, data.flatten())
    print "param_num = %d" % param_num


def write_parameter(outfile, feats):
    """
    :param outfile:
    :param feats:
    :return:
    """
    version = 0
    value_size = 4
    ret = ""
    for feat in feats:
        ret += feat.tostring()
    size = len(ret) / 4
    fo = open(outfile, 'wb')
    fo.write(struct.pack('iIQ', version, value_size, size))
    fo.write(ret)


def load_parameter(file):
    """
    loading
    """
    with open(file, 'rb') as f:
        f.read(16)
        return np.fromfile(f, dtype=np.float32)


if __name__ == "__main__":
    model_proto = sys.argv[1]
    paddle_model = sys.argv[2]
    conf = ModelConfig('%s.prototxt' % model_proto, '%s.caffemodel' % model_proto)
    load_model_parameters(conf, [], paddle_model)
    os.system("./caffe2mdl %s.prototxt %s.model" % (model_proto, model_proto))
    pass
