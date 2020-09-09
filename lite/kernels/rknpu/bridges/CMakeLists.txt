if(NOT LITE_WITH_RKNPU)
  return()
endif()

lite_cc_library(subgraph_bridge_utility_rknpu SRCS utility.cc DEPS ${rknpu_builder_libs} tensor)
lite_cc_library(subgraph_bridge_graph_rknpu SRCS graph.cc DEPS subgraph_bridge_utility_rknpu)

set(rknpu_subgraph_bridge_deps subgraph_bridge_registry subgraph_bridge_utility_rknpu subgraph_bridge_graph_rknpu)

lite_cc_library(subgraph_bridge_conv_op_rknpu SRCS conv_op.cc DEPS ${rknpu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_act_op_rknpu SRCS act_op.cc DEPS ${rknpu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_softmax_op_rknpu SRCS softmax_op.cc DEPS ${rknpu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_pool_op_rknpu SRCS pool_op.cc DEPS ${rknpu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_fc_op_rknpu SRCS fc_op.cc DEPS ${rknpu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_batch_norm_op_rknpu SRCS batch_norm_op.cc DEPS ${rknpu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_concat_op_rknpu SRCS concat_op.cc DEPS ${rknpu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_elementwise_ops_rknpu SRCS elementwise_ops.cc DEPS ${rknpu_subgraph_bridge_deps})


set(rknpu_subgraph_bridges
        subgraph_bridge_registry
        subgraph_bridge_utility_rknpu
        subgraph_bridge_graph_rknpu
        subgraph_bridge_conv_op_rknpu
        subgraph_bridge_act_op_rknpu
        subgraph_bridge_softmax_op_rknpu
        subgraph_bridge_pool_op_rknpu
        subgraph_bridge_fc_op_rknpu
        subgraph_bridge_batch_norm_op_rknpu
        subgraph_bridge_concat_op_rknpu
        subgraph_bridge_elementwise_ops_rknpu
        CACHE INTERNAL "rknpu_subgraph_bridges")

message(STATUS "+++++ rknpu_subgraph_bridges: ${rknpu_subgraph_bridges}")
