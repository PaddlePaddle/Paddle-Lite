if(NOT LITE_WITH_ARM)
    return()
endif()
set(lite_cv_deps)
lite_cc_library(cv_arm SRCS
            image_convert.cc
            image_preprocess.cc
            image2tensor.cc
            image_flip.cc
            image_rotate.cc
            image_transform.cc
            DEPS ${lite_cv_deps} tensor)

#if((NOT LITE_WITH_OPENCL AND NOT LITE_WITH_FPGA) AND (LITE_WITH_X86 OR LITE_WITH_ARM))
#    lite_cc_test(test_image_convert SRCS image_convert_test.cc DEPS image_convert_test)
#    lite_cc_test(test_image_preprocess SRCS image_preprocess_test.cc DEPS image_convert_test)
#    lite_cc_test(test_image2tensor SRCS image2tensor_test.cc DEPS image2tensor_test)
#endif()
