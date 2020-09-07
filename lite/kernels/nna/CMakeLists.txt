add_subdirectory(bridges)

add_kernel(subgraph_compute_nna NNA basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} device_nna subgraph_bridge_engine ${nna_subgraph_bridges})
