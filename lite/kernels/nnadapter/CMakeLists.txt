add_subdirectory(bridges)
aux_source_directory(converter CONVERTERS)

add_kernel(subgraph_compute_nnadapter NNADAPTER basic SRCS utility.cc ${CONVERTERS} engine.cc subgraph_compute.cc DEPS ${lite_kernel_deps} nnadapter_wrapper ${nnadapter_subgraph_bridges})
