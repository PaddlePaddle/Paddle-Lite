if(NOT LITE_WITH_CUDA)
    return()
endif()

get_property(cuda_static_deps GLOBAL PROPERTY CUDA_MODULES)

nv_library(cuda_activation SRCS activation.cu DEPS ${cuda_static_deps})
nv_library(cuda_scale SRCS scale.cu DEPS ${cuda_static_deps})
nv_library(cuda_type_trans SRCS type_trans.cu DEPS ${cuda_static_deps}) 
nv_library(cuda_transpose SRCS transpose.cu DEPS ${cuda_static_deps})
nv_library(cudnn_conv SRCS cudnn_conv.cc DEPS cuda_activation cuda_scale cuda_type_trans ${cuda_static_deps})
nv_library(cuda_elementwise SRCS elementwise.cu DEPS ${cuda_static_deps})
nv_library(cudnn_pool SRCS cudnn_pool.cc DEPS ${cuda_static_deps})
nv_library(cuda_gemm SRCS gemm.cc  DEPS ${cuda_static_deps})
nv_library(cuda_batched_gemm SRCS batched_gemm.cc DEPS ${cuda_static_deps})

set (
 math_cuda
 cudnn_conv
 cuda_activation
 cuda_scale
 cuda_type_trans
 cuda_transpose
 cuda_elementwise
 cudnn_pool
 cuda_gemm
 cuda_batched_gemm
)

set(math_cuda "${math_cuda}" CACHE GLOBAL "math cuda")
