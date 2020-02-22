if ((NOT LITE_ON_MODEL_OPTIMIZE_TOOL) AND (NOT LITE_WITH_OPENCL))
    return ()
endif()

set(cl_kernel_deps op_params cl_runtime cl_context cl_wrapper cl_target_wrapper cl_image_converter)

# image
add_kernel(mul_opencl OPENCL basic SRCS mul_compute.cc DEPS ${cl_kernel_deps})
add_kernel(elementwise_add_opencl OPENCL basic SRCS elementwise_add_compute.cc DEPS ${cl_kernel_deps})
add_kernel(elementwise_mul_opencl OPENCL basic SRCS elementwise_mul_compute.cc DEPS ${cl_kernel_deps})
add_kernel(fusion_elementwise_add_activation_opencl
           OPENCL basic SRCS fusion_elementwise_add_activation_compute.cc
           DEPS elementwise_add_opencl ${cl_kernel_deps})
add_kernel(pool_opencl OPENCL basic SRCS pool_image_compute.cc DEPS ${cl_kernel_deps})
add_kernel(io_copy_compute_opencl OPENCL basic SRCS io_copy_compute.cc DEPS ${tensor_lite} ${cl_kernel_deps})
add_kernel(activation_opencl OPENCL basic SRCS activation_image_compute.cc DEPS ${cl_kernel_deps})
add_kernel(reshape_opencl OPENCL basic SRCS reshape_image_compute.cc DEPS ${cl_kernel_deps})
add_kernel(conv_opencl OPENCL basic SRCS conv_image_compute.cc DEPS ${cl_kernel_deps} cl_image_converter)
add_kernel(layout_opencl OPENCL basic SRCS layout_compute.cc DEPS ${cl_kernel_deps})
add_kernel(concat_opencl OPENCL basic SRCS concat_image_compute.cc DEPS ${cl_kernel_deps})
add_kernel(nearest_interp_opencl OPENCL basic SRCS nearest_interp_image_compute.cc DEPS ${cl_kernel_deps})
add_kernel(scale_opencl OPENCL basic SRCS scale_image_compute.cc DEPS ${cl_kernel_deps})




# image test
lite_cc_test(test_activation_image_opencl SRCS activation_image_compute_test.cc
             DEPS activation_opencl layout_opencl op_registry program context
             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_conv_image_opencl SRCS conv_image_compute_test.cc
             DEPS conv_opencl op_registry program context cl_image_converter
             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_depthwise_conv2d_image_opencl SRCS depthwise_conv2d_image_compute_test.cc
             DEPS conv_opencl op_registry program context
             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_nearest_interp_image_opencl SRCS nearest_interp_image_compute_test.cc
             DEPS nearest_interp_opencl layout_opencl op_registry program context cl_image_converter
             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_pool_image_opencl SRCS pool_image_compute_test.cc
             DEPS pool_opencl op_registry program context
             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_scale_image_opencl SRCS scale_image_compute_test.cc
             DEPS scale_opencl op_registry program context
             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_reshape_image_opencl SRCS reshape_image_compute_test.cc
             DEPS reshape_opencl op_registry program context
             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_concat_image_opencl SRCS concat_image_compute_test.cc
             DEPS concat_opencl layout_opencl op_registry program context
             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)





# buffer
#add_kernel(activation_opencl OPENCL basic SRCS activation_buffer_compute.cc DEPS ${cl_kernel_deps})
#add_kernel(conv_opencl OPENCL basic SRCS conv_buffer_compute.cc DEPS ${cl_kernel_deps} cl_image_converter)
#add_kernel(depthwise_conv2d_opencl OPENCL basic SRCS depthwise_conv2d_buffer_compute.cc DEPS ${cl_kernel_deps})
#add_kernel(pool_opencl OPENCL basic SRCS pool_buffer_compute.cc DEPS ${cl_kernel_deps})
#add_kernel(concat_opencl OPENCL basic SRCS concat_buffer_compute.cc DEPS ${cl_kernel_deps})
add_kernel(fc_opencl OPENCL basic SRCS fc_buffer_compute.cc DEPS ${cl_kernel_deps})


# buffer test
#lite_cc_test(test_activation_buffer_opencl SRCS activation_buffer_compute_test.cc
#             DEPS activation_opencl op_registry program context
#             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

#lite_cc_test(test_conv_buffer_opencl SRCS conv_buffer_compute_test.cc
#             DEPS conv_opencl op_registry program context
#             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

#lite_cc_test(test_depthwise_conv2d_buffer_opencl SRCS depthwise_conv2d_buffer_compute_test.cc
#             DEPS depthwise_conv2d_opencl op_registry program context
#             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

#lite_cc_test(test_pool_buffer_opencl SRCS pool_buffer_compute_test.cc
#             DEPS pool_opencl op_registry program context
#             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

#lite_cc_test(test_concat_buffer_opencl SRCS concat_buffer_compute_test.cc
#             DEPS concat_opencl op_registry program context
#             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_fc_buffer_opencl SRCS fc_buffer_compute_test.cc
             DEPS fc_opencl op_registry program context
             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)







lite_cc_test(test_elementwise_add_opencl SRCS elementwise_add_compute_test.cc
             DEPS elementwise_add_opencl fusion_elementwise_add_activation_opencl op_registry program context
             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_elementwise_mul_opencl SRCS elementwise_mul_compute_test.cc
             DEPS elementwise_mul_opencl op_registry program context
             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

# TODO(ysh329): comment for buffer-impl mul
#lite_cc_test(test_mul_opencl SRCS mul_compute_test.cc
#        DEPS mul_opencl op_registry program context
#        ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

lite_cc_test(test_io_copy_compute_opencl SRCS io_copy_compute_test.cc
             DEPS io_copy_compute_opencl op_registry program context
             ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)





lite_cc_test(test_layout_opencl SRCS layout_compute_test.cc
        DEPS layout_opencl op_registry program context cl_image_converter
        ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)

         
