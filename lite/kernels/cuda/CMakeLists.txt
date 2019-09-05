if(NOT LITE_WITH_CUDA)
    return()
endif()

message(STATUS "compile with lite CUDA kernels")

nv_library(mul_compute_cuda SRCS mul_compute.cc DEPS ${lite_kernel_deps} context)
lite_cc_library(io_copy_compute_cuda SRCS io_copy_compute.cc DEPS ${lite_kernel_deps})
nv_library(leaky_relu_compute_cuda SRCS leaky_relu_compute.cu DEPS ${lite_kernel_deps})
nv_library(yolo_box_compute_cuda SRCS yolo_box_compute.cu DEPS ${lite_kernel_deps})
nv_library(nearest_interp_compute_cuda SRCS nearest_interp_compute.cu DEPS ${lite_kernel_deps})
nv_library(conv2d_cuda SRCS conv_compute.cc DEPS ${lite_kernel_deps}
${math_cuda})
nv_test(conv2d_cuda_test SRCS conv_compute_test.cc DEPS conv2d_cuda)

nv_test(nearest_interp_compute_cuda_test SRCS nearest_interp_compute_test.cc DEPS nearest_interp_compute_cuda)
nv_test(leaky_relu_compute_cuda_test SRCS leaky_relu_compute_test.cc DEPS leaky_relu_compute_cuda)
nv_test(yolo_box_compute_cuda_test SRCS yolo_box_compute_test.cc DEPS yolo_box_compute_cuda)

set(cuda_kernels
conv2d_cuda
mul_compute_cuda
io_copy_compute_cuda
leaky_relu_compute_cuda
nearest_interp_compute_cuda
yolo_box_compute_cuda
)

set(cuda_kernels "${cuda_kernels}" CACHE GLOBAL "cuda kernels")
