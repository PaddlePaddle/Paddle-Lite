if(NOT LITE_WITH_CUDA)
    return()
endif()
get_property(cuda_static_deps GLOBAL PROPERTY CUDA_STATIC_MODULES)

nv_library(target_wrapper_cuda SRCS target_wrapper.cc DEPS ${cuda_static_deps})
nv_library(cuda_blas SRCS blas.cc DEPS ${cuda_static_deps})
 
add_subdirectory(math)
