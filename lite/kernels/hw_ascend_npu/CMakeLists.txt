add_subdirectory(bridges)

add_kernel(subgraph_compute_hw_ascend_npu HW_ASCEND_NPU basic SRCS subgraph_compute.cc DEPS
  build_hw_ascend_npu
  device_hw_ascend_npu
  subgraph_bridge_engine
  runtime_hw_ascend_npu
  ${hw_ascend_npu_subgraph_bridges}
  subgraph_bridge_registry
  ${lite_kernel_deps}
  )
