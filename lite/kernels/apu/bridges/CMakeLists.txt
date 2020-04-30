if(NOT LITE_WITH_APU)
  return()
endif()


lite_cc_library(subgraph_bridge_utility_apu SRCS utility.cc DEPS tensor neuron_adapter)
lite_cc_library(subgraph_bridge_graph_apu SRCS graph.cc DEPS subgraph_bridge_utility_apu)

set(apu_subgraph_bridge_deps subgraph_bridge_registry subgraph_bridge_utility_apu subgraph_bridge_graph_apu)

lite_cc_library(subgraph_bridge_conv_op_apu SRCS conv_op.cc DEPS ${apu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_elementwise_ops_apu SRCS elementwise_ops.cc DEPS ${apu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_act_op_apu SRCS act_op.cc DEPS ${apu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_pool_op_apu SRCS pool_op.cc DEPS ${apu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_softmax_op_apu SRCS softmax_op.cc DEPS ${apu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_fc_op_apu SRCS fc_op.cc DEPS ${apu_subgraph_bridge_deps})


set(apu_subgraph_bridges
        subgraph_bridge_registry
        subgraph_bridge_utility_apu
        subgraph_bridge_conv_op_apu
        subgraph_bridge_elementwise_ops_apu
        subgraph_bridge_act_op_apu
        subgraph_bridge_softmax_op_apu
        subgraph_bridge_fc_op_apu
        subgraph_bridge_pool_op_apu
        CACHE INTERNAL "apu_subgraph_bridges")

message(STATUS "+++++ apu_subgraph_bridges: ${apu_subgraph_bridges}")
