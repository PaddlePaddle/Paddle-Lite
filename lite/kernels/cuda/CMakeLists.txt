if(NOT LITE_WITH_CUDA)
    return()
endif()

message(STATUS "compile with lite CUDA kernels")

nv_library(mul_compute_cuda SRCS mul_compute.cc DEPS ${lite_kernel_deps} context)

lite_cc_library(io_copy_compute_cuda SRCS io_copy_compute.cc DEPS ${lite_kernel_deps})

nv_library(leaky_relu_compute_cuda SRCS leaky_relu_compute.cu DEPS ${lite_kernel_deps})
lite_cc_test(leaky_relu_compute_cuda_test SRCS leaky_relu_compute_test.cc DEPS leaky_relu_compute_cuda)

nv_library(nearest_interp_compute_cuda SRCS nearest_interp_compute.cu DEPS ${lite_kernel_deps})
lite_cc_test(nearest_interp_compute_cuda_test SRCS nearest_interp_compute_test.cc DEPS nearest_interp_compute_cuda)

nv_library(concat_compute_cuda SRCS concat_compute.cu DEPS ${lite_kernel_deps})
lite_cc_test(concat_compute_cuda_test SRCS concat_compute_test.cc DEPS concat_compute_cuda)

nv_library(elementwise_add_compute_cuda SRCS elementwise_add_compute.cu DEPS ${lite_kernel_deps})
lite_cc_test(elementwise_add_compute_cuda_test SRCS elementwise_add_compute_test.cc DEPS elementwise_add_compute_cuda)

set(cuda_kernels
mul_compute_cuda
io_copy_compute_cuda
leaky_relu_compute_cuda
nearest_interp_compute_cuda
concat_compute_cuda
elementwise_add_compute_cuda
)

set(cuda_kernels "${cuda_kernels}" CACHE GLOBAL "cuda kernels")
