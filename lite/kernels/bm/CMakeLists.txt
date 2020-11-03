if(NOT LITE_WITH_BM)
  return ()
endif()

if(BM_DYNAMIC_COMPILE)
  add_definitions(-DBM_DYNAMIC_COMPILE)
endif()

add_subdirectory(bridges)
add_kernel(subgraph_compute_bm BM basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} subgraph_engine_base ${bm_subgraph_bridges})
