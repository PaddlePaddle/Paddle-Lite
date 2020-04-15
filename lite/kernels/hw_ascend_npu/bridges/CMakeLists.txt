if (NOT LITE_WITH_HW_ASCEND_NPU)
  return()
endif()

message(STATUS "======compile hw_ascend_npu bridges, ${hw_ascend_npu_builder_libs}")

lite_cc_library(subgraph_bridge_utility_hw_ascend_npu
  SRCS utility.cc
  DEPS ${hw_ascend_npu_builder_libs} tensor)

lite_cc_library(subgraph_bridge_graph_hw_ascend_npu
  SRCS graph.cc
  DEPS ${hw_ascend_npu_builder_libs} subgraph_bridge_utility_hw_ascend_npu)

set(hw_ascend_npu_subgraph_bridges
  subgraph_bridge_graph_hw_ascend_npu
  CACHE INTERNAL "hw_ascend_npu_subgraph_bridges")
