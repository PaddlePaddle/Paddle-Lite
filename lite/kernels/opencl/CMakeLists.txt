if (NOT LITE_WITH_OPENCL)
    return ()
endif()

set(cl_kernel_deps op_params cl_runtime cl_context cl_wrapper cl_target_wrapper)

add_kernel(fc_opencl OPENCL basic SRCS fc_compute.cc DEPS ${cl_kernel_deps})
add_kernel(mul_opencl OPENCL basic SRCS mul_compute.cc DEPS ${cl_kernel_deps})
add_kernel(elementwise_add_opencl OPENCL basic SRCS elementwise_add_compute.cc DEPS ${cl_kernel_deps})
add_kernel(fusion_elementwise_add_activation_opencl
        OPENCL basic SRCS fusion_elementwise_add_activation_compute.cc
        DEPS elementwise_add_opencl ${cl_kernel_deps})
add_kernel(pool_opencl OPENCL basic SRCS pool_compute.cc DEPS ${cl_kernel_deps})
add_kernel(io_copy_compute_opencl OPENCL basic SRCS io_copy_compute.cc DEPS ${tensor_lite} ${cl_kernel_deps})
add_kernel(relu_opencl OPENCL basic SRCS relu_compute.cc DEPS ${cl_kernel_deps})
add_kernel(depthwise_conv2d_opencl OPENCL basic SRCS depthwise_conv2d_compute.cc DEPS ${cl_kernel_deps})
add_kernel(conv_opencl OPENCL basic SRCS conv_compute.cc DEPS ${cl_kernel_deps})

lite_cc_test(test_elementwise_add_opencl SRCS elementwise_add_compute_test.cc
        DEPS elementwise_add_opencl fusion_elementwise_add_activation_opencl op_registry program context
        ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_pool_opencl SRCS pool_compute_test.cc
        DEPS pool_opencl op_registry program context
        ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_fc_opencl SRCS fc_compute_test.cc
        DEPS fc_opencl op_registry program context
        ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_mul_opencl SRCS mul_compute_test.cc
        DEPS mul_opencl op_registry program context
        ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_io_copy_compute_opencl SRCS io_copy_compute_test.cc
        DEPS io_copy_compute_opencl op_registry program context
        ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_relu_opencl SRCS relu_compute_test.cc
        DEPS relu_opencl op_registry program context
        ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_depthwise_conv2d_opencl SRCS depthwise_conv2d_compute_test.cc
        DEPS depthwise_conv2d_opencl op_registry program context
        ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_conv_opencl SRCS conv_compute_test.cc
        DEPS conv_opencl op_registry program context
        ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)
