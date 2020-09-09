add_subdirectory(bridges)

add_kernel(subgraph_compute_rknpu RKNPU basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} device_rknpu subgraph_engine_base ${rknpu_subgraph_bridges})
