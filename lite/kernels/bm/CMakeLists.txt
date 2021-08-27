add_subdirectory(bridges)

if(NOT LITE_WITH_BM)
  return ()
endif()

if(BM_DYNAMIC_COMPILE)
  add_definitions(-DBM_DYNAMIC_COMPILE)
endif()

if(BM_SAVE_UMODEL)
  add_definitions(-DBM_SAVE_UMODEL)
endif()

if(BM_SAVE_BMODEL)
  add_definitions(-DBM_SAVE_BMODEL)
endif()

add_kernel(subgraph_compute_bm BM basic SRCS subgraph_compute.cc)
set(lite_kernel_deps ${bm_subgraph_bridges})
