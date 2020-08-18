if(NOT LITE_WITH_CUDA)
    return()
endif()

get_property(cuda_static_deps GLOBAL PROPERTY CUDA_MODULES)

nv_library(cuda_activation SRCS activation.cu DEPS ${cuda_static_deps})
nv_library(cuda_scale SRCS scale.cu DEPS ${cuda_static_deps})
nv_library(cuda_type_trans SRCS type_trans.cu DEPS ${cuda_static_deps}) 
nv_library(cuda_transpose SRCS transpose.cu DEPS ${cuda_static_deps})
nv_library(cudnn_helper SRCS cudnn_helper.cc DEPS ${cuda_static_deps})
nv_library(cudnn_conv SRCS cudnn_conv.cc DEPS cuda_activation cuda_scale cuda_type_trans cudnn_helper ${cuda_static_deps})
nv_library(cudnn_softmax SRCS cudnn_softmax.cc DEPS cudnn_helper ${cuda_static_deps} tensor)
nv_library(cuda_elementwise SRCS elementwise.cu DEPS ${cuda_static_deps})
nv_library(cudnn_pool SRCS cudnn_pool.cc DEPS ${cuda_static_deps})
nv_library(cuda_gru_forward  SRCS gru_forward.cu DEPS cuda_activation ${cuda_static_deps})
nv_library(cuda_sequence2batch  SRCS sequence2batch.cu DEPS ${cuda_static_deps})
nv_library(cuda_gemm SRCS gemm.cc  DEPS ${cuda_static_deps})
nv_library(cuda_batched_gemm SRCS batched_gemm.cc DEPS ${cuda_static_deps})
nv_library(cuda_strided_gemm SRCS strided_gemm.cc DEPS ${cuda_static_deps})
nv_library(cuda_sequence_padding SRCS sequence_padding.cu DEPS ${cuda_static_deps})
nv_library(cuda_bias SRCS bias.cu DEPS ${cuda_static_deps})
nv_library(cuda_sequence_helper SRCS sequence_helper.cu DEPS ${cuda_static_deps})

set (
 math_cuda
 cudnn_conv
 cudnn_softmax
 cuda_activation
 cuda_scale
 cuda_type_trans
 cuda_transpose
 cuda_elementwise
 cudnn_pool
 cuda_gru_forward
 cuda_sequence2batch
 cuda_gemm
 cuda_batched_gemm
 cuda_strided_gemm
 cuda_sequence_padding
 cuda_bias
 cudnn_helper
 cuda_sequence_helper
)

set(math_cuda "${math_cuda}" CACHE GLOBAL "math cuda")
