if(NOT LITE_WITH_NNADAPTER)
  return()
endif()

lite_cc_library(subgraph_bridge_utility_nnadapter SRCS utility.cc DEPS core)
lite_cc_library(subgraph_bridge_converter_nnadapter SRCS converter.cc DEPS nnadapter_wrapper subgraph_bridge_utility_nnadapter)

set(nnadapter_subgraph_bridge_deps core subgraph_bridge_utility_nnadapter subgraph_bridge_converter_nnadapter)

lite_cc_library(subgraph_bridge_fc_op_nnadapter SRCS fc_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_cast_op_nnadapter SRCS cast_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_norm_op_nnadapter SRCS norm_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_dropout_op_nnadapter SRCS dropout_op.cc DEPS ${nnadapter_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_pad_op_nnadapter SRCS pad_op.cc DEPS ${nnadapter_subgraph_bridge_deps})

set(nnadapter_subgraph_bridges
        core
        subgraph_bridge_utility_nnadapter
        subgraph_bridge_converter_nnadapter
        subgraph_bridge_fc_op_nnadapter
        subgraph_bridge_cast_op_nnadapter
        subgraph_bridge_norm_op_nnadapter
        subgraph_bridge_dropout_op_nnadapter
        subgraph_bridge_pad_op_nnadapter
        CACHE INTERNAL "nnadapter_subgraph_bridges")

message(STATUS "+++++ nnadapter_subgraph_bridges: ${nnadapter_subgraph_bridges}")
