if(NOT LITE_WITH_BM)
  return ()
endif()

add_kernel(conv_compute_bm BM basic SRCS conv_compute.cc DEPS ${lite_kernel_deps})
add_kernel(calib_compute_bm BM basic SRCS calib_compute.cc DEPS ${lite_kernel_deps})
add_kernel(pool_compute_bm BM basic SRCS pool_compute.cc DEPS ${lite_kernel_deps})
add_kernel(elementwise_compute_bm BM basic SRCS elementwise_compute.cc DEPS ${lite_kernel_deps})
add_kernel(relu_compute_bm BM basic SRCS relu_compute.cc DEPS ${lite_kernel_deps})
add_kernel(softmax_compute_bm BM basic SRCS softmax_compute.cc DEPS ${lite_kernel_deps})
add_kernel(batch_norm_compute_bm BM basic SRCS batch_norm_compute.cc DEPS ${lite_kernel_deps})
add_kernel(scale_compute_bm BM basic SRCS scale_compute.cc DEPS ${lite_kernel_deps})
add_kernel(mul_compute_bm BM basic SRCS mul_compute.cc DEPS ${lite_kernel_deps})
add_kernel(io_copy_compute_bm BM basic SRCS io_copy_compute.cc DEPS ${lite_kernel_deps})

message(STATUS "compile with lite BM kernels")

