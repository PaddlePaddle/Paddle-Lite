if (NOT LITE_WITH_OPENCL)
    return()
endif()

lite_cc_library(cl_wrapper SRCS cl_wrapper.cc)
lite_cc_library(cl_utility SRCS cl_utility.cc DEPS cl_wrapper)
lite_cc_library(cl_runtime SRCS cl_runtime.cc DEPS cl_utility)
lite_cc_library(cl_context SRCS cl_context.cc DEPS cl_runtime)
lite_cc_library(cl_image_converter SRCS cl_image_converter.cc DEPS tensor)
lite_cc_library(cl_image SRCS cl_image.cc DEPS tensor cl_image_converter cl_runtime)
lite_cc_library(cl_caller SRCS cl_caller.cc  DEPS cl_context cl_image)
lite_cc_library(cl_target_wrapper SRCS target_wrapper.cc DEPS cl_runtime)
lite_cc_test(test_cl_functions SRCS cl_functions_test.cc DEPS cl_context cl_image cl_caller cl_wrapper cl_target_wrapper
             ARGS --cl_path=${CMAKE_SOURCE_DIR}/paddle/fluid/lite/backends/opencl)
lite_cc_test(test_cl_im2col SRCS cl_im2col_test.cc DEPS tensor cl_context cl_wrapper cl_target_wrapper
             ARGS --cl_path=${CMAKE_SOURCE_DIR}/paddle/fluid/lite/backends/opencl)

add_dependencies(cl_wrapper opencl_clhpp)
