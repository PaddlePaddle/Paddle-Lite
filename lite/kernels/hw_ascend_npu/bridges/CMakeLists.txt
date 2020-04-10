if (NOT LITE_WITH_HW_ASCEND_NPU)
  return()
endif()

message(STATUS "======compile hw_ascend_npu bridges, ${ascend_builder_libs}")

lite_cc_library(subgraph_bridge_graph_hw_ascend_npu
  SRCS graph.cc
  DEPS ${ascend_builder_libs})

set(hw_ascend_npu_subgraph_bridges
  subgraph_bridge_graph_hw_ascend_npu
  CACHE INTERNAL "hw_ascend_npu_subgraph_bridges")
