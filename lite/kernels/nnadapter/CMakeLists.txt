if(LITE_WITH_NNADAPTER)
  set(IS_FAKED_KERNEL false CACHE INTERNAL "")
elseif(LITE_ON_MODEL_OPTIMIZE_TOOL OR LITE_WITH_PYTHON)
  set(IS_FAKED_KERNEL true CACHE INTERNAL "")
else()
  return()
endif()


add_subdirectory(bridges)
aux_source_directory(converter CONVERTERS)

add_kernel(subgraph_compute_nnadapter NNADAPTER basic SRCS utility.cc ${CONVERTERS} engine.cc subgraph_compute.cc)

set(lite_kernel_deps ${lite_kernel_deps} ${nnadapter_subgraph_bridges})
