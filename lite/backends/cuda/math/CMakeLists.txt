if(NOT LITE_WITH_CUDA)
    return()
endif()

set(cuda_static_deps cudnn_static cublas_static curand_static
    culibos_static cudart_static)

nv_library(cuda_activation SRCS activation.cu DEPS ${cuda_static_deps})
nv_library(cuda_scale SRCS scale.cu DEPS ${cuda_static_deps})
nv_library(cuda_type_trans SRCS type_trans.cu DEPS ${cuda_static_deps}) 
nv_library(cuda_transpose SRCS transpose.cu DEPS ${cuda_static_deps})
nv_library(cudnn_conv SRCS cudnn_conv.cc DEPS cuda_activation cuda_scale
cuda_type_trans ${cuda_static_deps})
nv_library(cuda_elementwise SRCS elementwise.cu DEPS ${cuda_static_deps})
nv_library(cudnn_pool SRCS cudnn_pool.cc DEPS ${cuda_static_deps})

set (
 math_cuda
 cudnn_conv
 cuda_activation
 cuda_scale
 cuda_type_trans
 cuda_transpose
 cuda_elementwise
 cudnn_pool
)

set(math_cuda "${math_cuda}" CACHE GLOBAL "math cuda")
