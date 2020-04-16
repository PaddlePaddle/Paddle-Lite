if (NOT LITE_WITH_HW_ASCEND_NPU)
  return()
endif()

lite_cc_library(subgraph_bridge_utility_hw_ascend_npu
  SRCS utility.cc
  DEPS ${hw_ascend_npu_builder_libs} tensor)

lite_cc_library(subgraph_bridge_graph_hw_ascend_npu
  SRCS graph.cc
  DEPS ${hw_ascend_npu_builder_libs} subgraph_bridge_utility_hw_ascend_npu
  )

set(hw_ascend_npu_subgraph_bridge_deps subgraph_bridge_registry subgraph_bridge_utility_hw_ascend_npu subgraph_bridge_graph_hw_ascend_npu)

lite_cc_library(subgraph_bridge_act_op_hw_ascend_npu SRCS act_op.cc DEPS
  ${hw_ascend_npu_subgraph_bridge_deps}
  ${hw_ascend_npu_builder_libs})

set(hw_ascend_npu_subgraph_bridges
  subgraph_bridge_graph_hw_ascend_npu
  subgraph_bridge_registry
  subgraph_bridge_utility_hw_ascend_npu
  subgraph_bridge_act_op_hw_ascend_npu
  CACHE INTERNAL "hw_ascend_npu_subgraph_bridges")
