if(NOT LITE_WITH_BM)
  return ()
endif()

add_kernel(conv_2d_bm BM basic SRCS conv_compute.cc DEPS ${lite_kernel_deps})
add_kernel(calib_compute_bm BM basic SRCS calib_compute.cc DEPS ${lite_kernel_deps})

message(STATUS "compile with lite BM kernels")

