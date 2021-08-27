if(LITE_WITH_RKNPU)
  set(IS_FAKED_KERNEL false CACHE INTERNAL "")
elseif(LITE_ON_MODEL_OPTIMIZE_TOOL OR LITE_WITH_PYTHON)
  set(IS_FAKED_KERNEL true CACHE INTERNAL "")
else()
  return()
endif()


add_subdirectory(bridges)

add_kernel(subgraph_compute_rknpu RKNPU basic SRCS subgraph_compute.cc)
add_dependencies(kernels device_rknpu ${rknpu_subgraph_bridges})
target_link_libraries(kernels device_rknpu ${rknpu_subgraph_bridges})
