if(NOT LITE_WITH_CUDA)
    return()
endif()
get_property(cuda_deps GLOBAL PROPERTY CUDA_MODULES)

nv_library(target_wrapper_cuda SRCS target_wrapper.cc DEPS ${cuda_deps})
nv_library(cuda_blas SRCS blas.cc DEPS ${cuda_deps})
 
add_subdirectory(math)
