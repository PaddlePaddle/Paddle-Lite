add_subdirectory(bridges)

add_kernel(subgraph_compute_npu NPU basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} device_npu subgraph_engine_base ${npu_subgraph_bridges})
