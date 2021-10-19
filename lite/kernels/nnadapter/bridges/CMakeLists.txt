if(NOT LITE_WITH_NNADAPTER)
  return()
endif()

lite_cc_library(subgraph_bridge_utility_nnadapter SRCS utility.cc DEPS core)
lite_cc_library(subgraph_bridge_converter_nnadapter SRCS converter.cc DEPS nnadapter_wrapper subgraph_bridge_utility_nnadapter)

set(nnadapter_subgraph_bridge_deps core subgraph_bridge_utility_nnadapter subgraph_bridge_converter_nnadapter)

lite_cc_library(subgraph_bridge_fc_op_nnadapter SRCS fc_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_transpose_op_nnadapter SRCS transpose_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_cast_op_nnadapter SRCS cast_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_norm_op_nnadapter SRCS norm_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_assign_value_op_nnadapter SRCS assign_value_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_assign_op_nnadapter SRCS assign_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_pow_op_nnadapter SRCS pow_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_batch_normalization_op_nnadapter SRCS batch_normalization_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_clip_op_nnadapter SRCS clip_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_reduce_mean_op_nnadapter SRCS reduce_mean_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_dropout_op_nnadapter SRCS dropout_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_pad_op_nnadapter SRCS pad_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_stack_op_nnadapter SRCS stack_op.cc DEPS ${nnadapter_subgraph_bridge_deps})

set(nnadapter_subgraph_bridges
        core
        subgraph_bridge_utility_nnadapter
        subgraph_bridge_converter_nnadapter
        subgraph_bridge_fc_op_nnadapter
        subgraph_bridge_transpose_op_nnadapter
        subgraph_bridge_cast_op_nnadapter
        subgraph_bridge_norm_op_nnadapter
        subgraph_bridge_assign_op_nnadapter
        subgraph_bridge_assign_value_op_nnadapter
        subgraph_bridge_pow_op_nnadapter
        subgraph_bridge_batch_normalization_op_nnadapter
        subgraph_bridge_clip_op_nnadapter
        subgraph_bridge_reduce_mean_op_nnadapter
        subgraph_bridge_dropout_op_nnadapter
        subgraph_bridge_stack_op_nnadapter
        subgraph_bridge_pad_op_nnadapter
        CACHE INTERNAL "nnadapter_subgraph_bridges")

message(STATUS "+++++ nnadapter_subgraph_bridges: ${nnadapter_subgraph_bridges}")
