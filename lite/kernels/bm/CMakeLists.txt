if(LITE_WITH_BM)
  set(IS_FAKED_KERNEL false CACHE INTERNAL "")
  add_subdirectory(bridges)
  set(lite_kernel_deps ${lite_kernel_deps} ${bm_subgraph_bridges} CACHE INTERNAL "")
elseif(LITE_ON_MODEL_OPTIMIZE_TOOL OR LITE_WITH_PYTHON)
  set(IS_FAKED_KERNEL true CACHE INTERNAL "")
else()
  return()
endif()

# TODO: bm kernels info are not collected by opt
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
