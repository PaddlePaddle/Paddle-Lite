if(NOT LITE_WITH_XPU)
  return()
endif()

lite_cc_library(subgraph_bridge_utility_xpu SRCS utility.cc DEPS ${xpu_builder_libs} tensor)
lite_cc_library(subgraph_bridge_graph_xpu SRCS graph.cc DEPS subgraph_bridge_utility_xpu)

set(xpu_subgraph_bridge_deps subgraph_bridge_registry subgraph_bridge_utility_xpu subgraph_bridge_graph_xpu)

lite_cc_library(subgraph_bridge_act_op_xpu SRCS act_op.cc DEPS ${xpu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_conv_op_xpu SRCS conv_op.cc DEPS ${xpu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_elementwise_ops_xpu SRCS elementwise_ops.cc DEPS ${xpu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_pool_op_xpu SRCS pool_op.cc DEPS ${subgraph_bridge_deps_xpu})
lite_cc_library(subgraph_bridge_softmax_op_xpu SRCS softmax_op.cc DEPS ${subgraph_bridge_deps_xpu})
lite_cc_library(subgraph_bridge_mul_op_xpu SRCS mul_op.cc DEPS ${xpu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_batch_norm_op_xpu SRCS batch_norm_op.cc DEPS ${xpu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_transpose_op_xpu SRCS transpose_op.cc DEPS ${xpu_subgraph_bridge_deps})

set(xpu_subgraph_bridges
        subgraph_bridge_registry
        subgraph_bridge_utility_xpu
        subgraph_bridge_graph_xpu
        subgraph_bridge_act_op_xpu
        subgraph_bridge_conv_op_xpu
        subgraph_bridge_elementwise_ops_xpu
        subgraph_bridge_pool_op_xpu
        subgraph_bridge_softmax_op_xpu
        subgraph_bridge_mul_op_xpu
        subgraph_bridge_batch_norm_op_xpu
        subgraph_bridge_transpose_op_xpu
        CACHE INTERNAL "xpu_subgraph_bridges")

message(STATUS "+++++ xpu_subgraph_bridges: ${xpu_subgraph_bridges}")
