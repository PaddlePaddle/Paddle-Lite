add_subdirectory(bridges)

add_kernel(subgraph_compute_nnadapter NNADAPTER basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} subgraph_engine_base ${nnadapter_subgraph_bridges})
