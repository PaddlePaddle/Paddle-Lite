# coding=utf-8

# mdl layers
layer_mdl_conv = 'ConvolutionLayer'
layer_mdl_deepwise_conv = 'DepthwiseConvolutionLayer'
layer_mdl_relu = 'ReluLayer'
layer_mdl_pointwise_add = 'PointwiseConvolutionLayer'

# fluid ops
op_fluid_fusion_conv_add = 'fusion_conv_add'
op_fluid_relu = 'relu'

# dict mdk layer ---  fluid op
mdl2fluid_op_layer_dict = {
    layer_mdl_conv: op_fluid_fusion_conv_add,
    layer_mdl_deepwise_conv: op_fluid_fusion_conv_add,
    layer_mdl_relu: op_fluid_relu,
    layer_mdl_pointwise_add: op_fluid_fusion_conv_add
}

mdl_outputs_key = "outputs"
mdl_inputs_key = "inputs"
mdl_weight_key = "weights"
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
    mdl_weight_key: ()

}
# mdl layers  ---  fluid ops
op_io_dict = {
    'fusion_conv_add': fusion_conv_add_dict,
    'relu': relu_dict
}

# fluid attr key  ---  mdl params key
fusion_conv_add_attrs_dict = {
    'paddings': 'pad',
    'strides': 'stride',
    'groups': 'group'
}
# fluid attr key  ---  mdl params key
fluid_attrs_type_dict = {
    'paddings': 0,
    'strides': 6,
    'groups': 6
}

# '': "bias_term",    是不是要add   目前 yolo的模型都是 bias_term = 1


# attrs {
#       name: "axis"
#       type: INT
#       i: 1
#     }


# attrs_name = {
#     'name': "workspace_size_MB",
#     'type': 'INT',
#     'i': '4096'
# }
# attrs
# {
#     name: "data_format"
#     type: STRING
#     s: "AnyLayout"
# }
# attrs
# {
#     name: "use_mkldnn"
#     type: BOOLEAN
#     b: false
# }
# attrs
# {
#     name: "use_cudnn"
#     type: BOOLEAN
#     b: true
# }
# attrs
# {
#     name: "dilations"
#     type: INTS
#     ints: 1
#     ints: 1
# }
# attrs
# {
#     name: "groups"
#     type: INT
#     i: 1
# }
# attrs
# {
#     name: "paddings"
#     type: INTS
#     ints: 0
#     ints: 0
# }
# attrs
# {
#     name: "strides"
#     type: INTS
#     ints: 1
#     ints: 1
# }
