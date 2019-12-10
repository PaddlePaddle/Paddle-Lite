
add_subdirectory(bridges)

add_kernel(subgraph_compute_xpu XPU basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} subgraph_engine_xpu)
