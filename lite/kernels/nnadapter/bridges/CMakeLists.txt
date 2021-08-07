if(NOT LITE_WITH_NNADAPTER)
  return()
endif()

lite_cc_library(subgraph_bridge_utility_nnadapter SRCS utility.cc DEPS tensor)
lite_cc_library(subgraph_bridge_converter_nnadapter SRCS converter.cc DEPS nnadapter_wrapper subgraph_bridge_utility_nnadapter)

set(nnadapter_subgraph_bridge_deps subgraph_bridge_registry subgraph_bridge_utility_nnadapter subgraph_bridge_converter_nnadapter)

lite_cc_library(subgraph_bridge_conv_op_nnadapter SRCS conv_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_act_op_nnadapter SRCS act_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_softmax_op_nnadapter SRCS softmax_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_fill_constant_op_nnadapter SRCS fill_constant_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
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
lite_cc_library(subgraph_bridge_shape_op_nnadapter SRCS shape_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_norm_op_nnadapter SRCS norm_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_assign_value_op_nnadapter SRCS assign_value_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_assign_op_nnadapter SRCS assign_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_deformable_conv_nnadapter SRCS deformable_conv_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_conv_transpose_op_nnadapter SRCS conv_transpose_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_hard_swish_op_nnadapter SRCS hard_swish_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_hard_sigmoid_op_nnadapter SRCS hard_sigmoid_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_unsqueeze_op_nnadapter SRCS unsqueeze_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
set(nnadapter_subgraph_bridges
        subgraph_bridge_registry
        subgraph_bridge_utility_nnadapter
        subgraph_bridge_converter_nnadapter
        subgraph_bridge_conv_op_nnadapter
        subgraph_bridge_act_op_nnadapter
        subgraph_bridge_softmax_op_nnadapter
        subgraph_bridge_fill_constant_op_nnadapter
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
        subgraph_bridge_shape_op_nnadapter
        subgraph_bridge_norm_op_nnadapter
        subgraph_bridge_assign_op_nnadapter
        subgraph_bridge_assign_value_op_nnadapter
        subgraph_bridge_deformable_conv_nnadapter
        subgraph_bridge_conv_transpose_op_nnadapter
        subgraph_bridge_hard_swish_op_nnadapter
        subgraph_bridge_hard_sigmoid_op_nnadapter
        subgraph_bridge_unsqueeze_op_nnadapter
        CACHE INTERNAL "nnadapter_subgraph_bridges")

message(STATUS "+++++ nnadapter_subgraph_bridges: ${nnadapter_subgraph_bridges}")
