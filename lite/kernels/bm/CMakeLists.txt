if(NOT LITE_WITH_BM)
  return ()
endif()

add_kernel(subgraph_compute_bm BM basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} device_bm subgraph_bridge_engine ${bm_subgraph_bridges})

add_subdirectory(bridges)
