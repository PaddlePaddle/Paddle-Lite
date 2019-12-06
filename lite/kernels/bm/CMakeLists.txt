if(NOT LITE_WITH_BM)
  return ()
endif()

add_kernel(graph_compute_bm BM basic SRCS graph_compute.cc DEPS ${lite_kernel_deps} )

add_subdirectory(bridges)
