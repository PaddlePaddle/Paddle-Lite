if(NOT LITE_WITH_IMAGINATION_NNA)
  return()
endif()

lite_cc_library(subgraph_bridge_utility_imagination_nna SRCS utility.cc DEPS ${imagination_nna_builder_libs} ${imagination_nna_runtime_libs} tensor)
lite_cc_library(subgraph_bridge_graph_imagination_nna SRCS graph.cc DEPS subgraph_bridge_utility_imagination_nna)

set(imagination_nna_subgraph_bridge_deps subgraph_bridge_registry subgraph_bridge_utility_imagination_nna subgraph_bridge_graph_imagination_nna)

lite_cc_library(subgraph_bridge_fc_op_imagination_nna SRCS fc_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_conv_op_imagination_nna SRCS conv_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_act_op_imagination_nna SRCS act_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_pool_op_imagination_nna SRCS pool_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})

set(imagination_nna_subgraph_bridges
        subgraph_bridge_registry
        subgraph_bridge_utility_imagination_nna
        subgraph_bridge_graph_imagination_nna
        subgraph_bridge_fc_op_imagination_nna
        subgraph_bridge_conv_op_imagination_nna
        subgraph_bridge_act_op_imagination_nna
        subgraph_bridge_pool_op_imagination_nna
        CACHE INTERNAL "imagination_nna_subgraph_bridges")

message(STATUS "+++++ imagination_nna_subgraph_bridges: ${imagination_nna_subgraph_bridges}")
