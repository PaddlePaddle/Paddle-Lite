if(NOT LITE_WITH_NNADAPTER)
  return()
endif()

lite_cc_library(subgraph_bridge_utility_nnadapter SRCS utility.cc DEPS tensor)
lite_cc_library(subgraph_bridge_graph_nnadapter SRCS graph.cc DEPS nnadapter_wrapper subgraph_bridge_utility_nnadapter)

set(nnadapter_subgraph_bridge_deps subgraph_bridge_registry subgraph_bridge_utility_nnadapter subgraph_bridge_graph_nnadapter)

lite_cc_library(subgraph_bridge_conv_op_nnadapter SRCS conv_op.cc DEPS ${nnadapter_subgraph_bridge_deps})


set(nnadapter_subgraph_bridges
        subgraph_bridge_registry
        subgraph_bridge_utility_nnadapter
        subgraph_bridge_graph_nnadapter
        subgraph_bridge_conv_op_nnadapter
        CACHE INTERNAL "nnadapter_subgraph_bridges")

message(STATUS "+++++ nnadapter_subgraph_bridges: ${nnadapter_subgraph_bridges}")
