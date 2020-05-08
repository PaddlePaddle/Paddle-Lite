if(LITE_WITH_CV AND (NOT LITE_WITH_OPENCL AND NOT LITE_WITH_FPGA AND NOT LITE_WITH_MLU) AND LITE_WITH_ARM)
    lite_cc_test(image_convert_test SRCS image_convert_test.cc DEPS paddle_cv_arm)
endif()
