if(LITE_WITH_NPU)
  set(IS_FAKED_KERNEL false CACHE INTERNAL "")
  add_subdirectory(bridges)
  set(lite_kernel_deps ${lite_kernel_deps} ${npu_subgraph_bridges} device_npu CACHE INTERNAL "")
elseif(LITE_ON_MODEL_OPTIMIZE_TOOL OR LITE_WITH_PYTHON)
  set(IS_FAKED_KERNEL true CACHE INTERNAL "")
else()
  return()
endif()



add_kernel(subgraph_compute_npu NPU basic SRCS subgraph_compute.cc)
