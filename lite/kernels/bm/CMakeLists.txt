if(NOT LITE_WITH_BM)
  return ()
endif()

add_subdirectory(bridges)
add_kernel(subgraph_compute_bm BM basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} ${bm_subgraph_bridges})
