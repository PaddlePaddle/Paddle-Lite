if(LITE_WITH_HUAWEI_ASCEND_NPU)
  set(IS_FAKED_KERNEL false CACHE INTERNAL "")
elseif(LITE_ON_MODEL_OPTIMIZE_TOOL OR LITE_WITH_PYTHON)
  set(IS_FAKED_KERNEL true CACHE INTERNAL "")
else()
  return()
endif()

add_subdirectory(bridges)

add_kernel(subgraph_compute_huawei_ascend_npu HUAWEI_ASCEND_NPU basic SRCS subgraph_compute.cc)
add_dependencies(kernels device_huawei_ascend_npu ${huawei_ascend_npu_subgraph_bridges})
target_link_libraries(kernels device_huawei_ascend_npu ${huawei_ascend_npu_subgraph_bridges})
