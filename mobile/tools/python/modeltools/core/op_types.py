# coding=utf-8

# mdl layers
layer_mdl_conv = 'ConvolutionLayer'
layer_mdl_deepwise_conv = 'DepthwiseConvolutionLayer'
layer_mdl_relu = 'ReluLayer'
layer_mdl_pointwise_add = 'PointwiseConvolutionLayer'
layer_mdl_pooling = 'PoolingLayer'
layer_mdl_softmax = 'SoftmaxLayer'

# fluid ops
op_fluid_fusion_conv_add = 'fusion_conv_add'
op_fluid_relu = 'relu'
op_fluid_pooling = 'pool2d'
op_fluid_softmax = 'softmax'

# dict mdk layer ---  fluid op
mdl2fluid_op_layer_dict = {
    layer_mdl_conv: op_fluid_fusion_conv_add,
    layer_mdl_deepwise_conv: op_fluid_fusion_conv_add,
    layer_mdl_relu: op_fluid_relu,
    layer_mdl_pointwise_add: op_fluid_fusion_conv_add,
    layer_mdl_pooling: op_fluid_pooling,
    layer_mdl_softmax: op_fluid_softmax
}

mdl_outputs_key = "outputs"
mdl_inputs_key = "inputs"
mdl_weight_key = "weight"
mdl_attrs_key = "params"

# dict of mdl-input _out param  to fluid input out attrs
fusion_conv_add_dict = {
    mdl_inputs_key: 'Input',
    mdl_outputs_key: 'Out',
    mdl_weight_key: ('Filter', 'Y'),
    mdl_attrs_key: (
        # 'workspace_size_MB', 'use_mkldnn', 'use_cudnn', 'data_format','dilations',
        # dilations =  [1,1]
        'groups', 'paddings', 'strides'
        # 'axis'
    )
}

relu_dict = {
    mdl_inputs_key: 'X',
    mdl_outputs_key: 'Out',
    # mdl_weight_key: ()

}

pool2d_dict = {
    mdl_inputs_key: 'X',
    mdl_outputs_key: 'Out',
    # mdl_weight_key: (),
    mdl_attrs_key: ('pooling_type', 'global_pooling')

}

softmax_dict = {
    mdl_inputs_key: 'X',
    mdl_outputs_key: 'Out',
    mdl_weight_key: (),
    mdl_attrs_key: ()
}
# mdl layers  ---  fluid ops
op_io_dict = {
    'fusion_conv_add': fusion_conv_add_dict,
    'relu': relu_dict,
    'pool2d': pool2d_dict,
    'softmax': softmax_dict
}

# fluid attr key  ---  mdl params key
fusion_conv_add_attrs_dict = {
    'paddings': 'pad',
    'strides': 'stride',
    'groups': 'group'
}

# fluid attr key  ---  mdl params key
pool2d_attrs_dict = {
    'global_pooling': 'global_pooling',
    'pooling_type': 'type'
}


# fluid attr key  ---  mdl params key
fluid_attrs_type_dict = {
    'paddings': 0,
    'strides': 6,
    'groups': 6
}
