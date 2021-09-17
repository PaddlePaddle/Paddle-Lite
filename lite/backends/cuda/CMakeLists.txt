if(NOT LITE_WITH_CUDA)
    return()
endif()
get_property(cuda_deps GLOBAL PROPERTY CUDA_MODULES)

nv_library(target_wrapper_cuda SRCS target_wrapper.cc DEPS ${cuda_deps})
nv_library(cuda_blas SRCS blas.cc DEPS ${cuda_deps})

nv_library(nvtx_wrapper SRCS nvtx_wrapper DEPS ${cuda_deps})

lite_cc_library(cuda_context SRCS context.cc)
 
add_subdirectory(math)
