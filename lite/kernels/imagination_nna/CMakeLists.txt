add_subdirectory(bridges)

add_kernel(subgraph_compute_imagination_nna IMAGINATION_NNA basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} device_imagination_nna subgraph_engine_base ${imagination_nna_subgraph_bridges})
