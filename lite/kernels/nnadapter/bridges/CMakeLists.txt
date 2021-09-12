if(NOT LITE_WITH_NNADAPTER)
  return()
endif()

lite_cc_library(subgraph_bridge_utility_nnadapter SRCS utility.cc DEPS core)
lite_cc_library(subgraph_bridge_converter_nnadapter SRCS converter.cc DEPS nnadapter_wrapper subgraph_bridge_utility_nnadapter)

set(nnadapter_subgraph_bridge_deps core subgraph_bridge_utility_nnadapter subgraph_bridge_converter_nnadapter)

lite_cc_library(subgraph_bridge_conv_op_nnadapter SRCS conv_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_pool_op_nnadapter SRCS pool_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_fc_op_nnadapter SRCS fc_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_elementwise_ops_nnadapter SRCS elementwise_ops.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_scale_op_nnadapter SRCS scale_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_reshape_op_nnadapter SRCS reshape_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_transpose_op_nnadapter SRCS transpose_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_concat_op_nnadapter SRCS concat_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_flatten_op_nnadapter SRCS flatten_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_interpolate_op_nnadapter SRCS interpolate_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_split_op_nnadapter SRCS split_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_cast_op_nnadapter SRCS cast_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_norm_op_nnadapter SRCS norm_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_assign_value_op_nnadapter SRCS assign_value_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_assign_op_nnadapter SRCS assign_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_deformable_conv_nnadapter SRCS deformable_conv_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_conv_transpose_op_nnadapter SRCS conv_transpose_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_hard_swish_op_nnadapter SRCS hard_swish_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_hard_sigmoid_op_nnadapter SRCS hard_sigmoid_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_squeeze_op_nnadapter SRCS squeeze_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_unsqueeze_op_nnadapter SRCS unsqueeze_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_pow_op_nnadapter SRCS pow_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_batch_normalization_op_nnadapter SRCS batch_normalization_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_clip_op_nnadapter SRCS clip_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_slice_op_nnadapter SRCS slice_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_reduce_mean_op_nnadapter SRCS reduce_mean_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_dropout_op_nnadapter SRCS dropout_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_expand_op_nnadapter SRCS expand_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_range_op_nnadapter SRCS range_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_leaky_relu_op_nnadapter SRCS leaky_relu_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_pad_op_nnadapter SRCS pad_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_stack_op_nnadapter SRCS stack_op.cc DEPS ${nnadapter_subgraph_bridge_deps})

set(nnadapter_subgraph_bridges
        core
        subgraph_bridge_utility_nnadapter
        subgraph_bridge_converter_nnadapter
        subgraph_bridge_conv_op_nnadapter
        subgraph_bridge_pool_op_nnadapter
        subgraph_bridge_fc_op_nnadapter
        subgraph_bridge_elementwise_ops_nnadapter
        subgraph_bridge_scale_op_nnadapter
        subgraph_bridge_reshape_op_nnadapter
        subgraph_bridge_transpose_op_nnadapter
        subgraph_bridge_concat_op_nnadapter
        subgraph_bridge_flatten_op_nnadapter
        subgraph_bridge_interpolate_op_nnadapter
        subgraph_bridge_split_op_nnadapter
        subgraph_bridge_cast_op_nnadapter
        subgraph_bridge_norm_op_nnadapter
        subgraph_bridge_assign_op_nnadapter
        subgraph_bridge_assign_value_op_nnadapter
        subgraph_bridge_deformable_conv_nnadapter
        subgraph_bridge_conv_transpose_op_nnadapter
        subgraph_bridge_hard_swish_op_nnadapter
        subgraph_bridge_hard_sigmoid_op_nnadapter
        subgraph_bridge_squeeze_op_nnadapter
        subgraph_bridge_unsqueeze_op_nnadapter
        subgraph_bridge_pow_op_nnadapter
        subgraph_bridge_batch_normalization_op_nnadapter
        subgraph_bridge_clip_op_nnadapter
        subgraph_bridge_slice_op_nnadapter
        subgraph_bridge_reduce_mean_op_nnadapter
        subgraph_bridge_dropout_op_nnadapter
        subgraph_bridge_expand_op_nnadapter
        subgraph_bridge_range_op_nnadapter
        subgraph_bridge_leaky_relu_op_nnadapter
        subgraph_bridge_stack_op_nnadapter
        subgraph_bridge_pad_op_nnadapter
        CACHE INTERNAL "nnadapter_subgraph_bridges")

message(STATUS "+++++ nnadapter_subgraph_bridges: ${nnadapter_subgraph_bridges}")
