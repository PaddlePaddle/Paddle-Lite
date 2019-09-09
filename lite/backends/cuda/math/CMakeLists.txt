if(NOT LITE_WITH_CUDA)
    return()
endif()

nv_library(cuda_activation SRCS activation.cu)
nv_library(cuda_scale SRCS scale.cu)
nv_library(cuda_type_trans SRCS type_trans.cu)
nv_library(cudnn_conv SRCS cudnn_conv.cc DEPS cuda_activation cuda_scale
cuda_type_trans)

set (
 math_cuda
 cudnn_conv
 cuda_activation
 cuda_scale
 cuda_type_trans
)

set(math_cuda "${math_cuda}" CACHE GLOBAL "math cuda")
