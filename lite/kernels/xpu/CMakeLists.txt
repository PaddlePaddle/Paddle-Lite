
add_subdirectory(bridges)

add_kernel(subgraph_compute_xpu XPU basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} device_xpu subgraph_bridge_engine ${xpu_subgraph_bridges})
