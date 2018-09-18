mdl2fluid_op_layer_dict = {
    'ConvolutionLayer': 'fusion_conv_add',
    'DepthwiseConvolutionLayer': 'fusion_conv_add',
    'ReluLayer': 'relu',
    'PointwiseConvolutionLayer': 'fusion_conv_add'
}

mdl_outputs_key = "outputs"
mdl_inputs_key = "inputs"
mdl_weight_key = "weights"
# inputs_key = "inputs"

fusion_conv_add_dict = {
    mdl_inputs_key: 'Input',
    mdl_outputs_key: 'Out',
    mdl_weight_key: ('Filter', 'Y')
}

relu_dict = {
    mdl_inputs_key: 'X',
    mdl_outputs_key: 'Out',
    mdl_weight_key: ()

}
op_io_dict = {
    'fusion_conv_add': fusion_conv_add_dict,
    'relu': relu_dict

}
