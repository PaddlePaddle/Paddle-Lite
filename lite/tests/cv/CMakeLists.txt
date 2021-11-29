if(LITE_WITH_CV AND (NOT LITE_WITH_OPENCL AND NOT LITE_WITH_FPGA AND NOT LITE_WITH_MLU AND NOT LITE_WITH_NNADAPTER) AND LITE_WITH_ARM)
    lite_cc_test(image_convert_test SRCS image_convert_test.cc)
    lite_cc_test(image_profiler_test SRCS image_profiler_test.cc DEPS anakin_cv_arm)
endif()
