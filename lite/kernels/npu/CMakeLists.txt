if(LITE_WITH_NPU)
  set(IS_FAKED_KERNEL false CACHE INTERNAL "")
elseif(LITE_ON_MODEL_OPTIMIZE_TOOL OR LITE_WITH_PYTHON)
  set(IS_FAKED_KERNEL true CACHE INTERNAL "")
else()
  return()
endif()

add_subdirectory(bridges)

add_kernel(subgraph_compute_npu NPU basic SRCS subgraph_compute.cc)

set(lite_kernel_deps ${lite_kernel_deps} ${npu_subgraph_bridges})
