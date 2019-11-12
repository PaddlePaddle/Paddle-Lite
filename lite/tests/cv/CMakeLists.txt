if(LITE_WITH_CV AND (NOT LITE_WITH_OPENCL AND NOT LITE_WITH_FPGA) AND LITE_WITH_ARM)
    lite_cc_test(image_convert_test SRCS image_convert_test.cc DEPS paddle_cv_arm paddle_api_light ${lite_cv_deps} ${arm_kernels} ${lite_ops} ${host_kernels})
endif()
