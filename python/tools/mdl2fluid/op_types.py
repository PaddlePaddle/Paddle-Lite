mdl2fluid_op_layer_dict = {
    'ConvolutionLayer': 'fusion_conv_add',
    'DepthwiseConvolutionLayer': 'fusion_conv_add',
    'ReluLayer': 'relu',
    'PointwiseConvolutionLayer': 'fusion_conv_add'
}

fusion_conv_add_dict = {
    'inputs': 'Input',
    'outputs': 'Out'
}

relu_dict = {
    'inputs': 'X',
    'outputs': 'Out'
}
