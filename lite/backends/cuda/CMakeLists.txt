if(NOT LITE_WITH_CUDA)
    return()
endif()
get_property(cuda_deps GLOBAL PROPERTY CUDA_MODULES)

nv_library(target_wrapper_cuda SRCS target_wrapper.cc DEPS ${cuda_deps})
nv_library(cuda_blas SRCS blas.cc DEPS ${cuda_deps})

nv_library(nvtx_wrapper SRCS nvtx_wrapper DEPS ${cuda_deps})

lite_cc_library(cuda_context SRCS context.cc DEPS device_info)
lite_cc_library(stream_guard SRCS stream_guard.cc DEPS target_wrapper_cuda ${cuda_deps})

add_subdirectory(math)
