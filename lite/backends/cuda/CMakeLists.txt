if(NOT LITE_WITH_CUDA)
    return()
endif()
set(cuda_static_deps cudnn_static cublas_static curand_static
    culibos_static cudart_static)

nv_library(target_wrapper_cuda SRCS target_wrapper.cc DEPS ${cuda_static_deps})
nv_library(cuda_blas SRCS blas.cc DEPS ${cuda_static_deps})
 
add_subdirectory(math)
