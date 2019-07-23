if (NOT LITE_WITH_OPENCL)
    return ()
endif()

set(cl_kernel_deps op_params cl_runtime cl_context cl_wrapper cl_target_wrapper)

lite_cc_library(mul_opencl SRCS mul_compute.cc DEPS ${cl_kernel_deps})
lite_cc_library(elementwise_add_opencl SRCS elementwise_add_compute.cc DEPS ${cl_kernel_deps})
lite_cc_library(pool_opencl SRCS pool_compute.cc DEPS ${cl_kernel_deps})
lite_cc_library(io_copy_compute_opencl SRCS io_copy_compute.cc DEPS ${tensor_lite} ${cl_kernel_deps})

lite_cc_test(test_elementwise_add_opencl SRCS elementwise_add_compute_test.cc DEPS elementwise_add_opencl
  op_registry program
  context
  ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/opencl
  )

lite_cc_test(test_pool_opencl SRCS pool_compute_test.cc DEPS pool_opencl
  op_registry program
  context
  ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/opencl
  )

lite_cc_test(test_mul_opencl SRCS mul_compute_test.cc DEPS mul_opencl
  op_registry program
  context
  ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/opencl
  )

lite_cc_test(test_io_copy_compute_opencl SRCS io_copy_compute_test.cc DEPS io_copy_compute_opencl
        op_registry program
        context)

set(opencl_kernels
    elementwise_add_opencl
    mul_opencl
    io_copy_compute_opencl
    pool_opencl
    CACHE INTERNAL "opencl_kernels")
