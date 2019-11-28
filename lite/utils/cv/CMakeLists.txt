if(LITE_WITH_CV AND (NOT LITE_WITH_OPENCL AND NOT LITE_WITH_FPGA) AND LITE_WITH_ARM)
    set(lite_cv_deps)
    lite_cc_library(paddle_cv_arm SRCS
            image_convert.cc
            paddle_image_preprocess.cc
            image2tensor.cc
            image_flip.cc
            image_rotate.cc
            image_resize.cc
            DEPS ${lite_cv_deps} paddle_api_light)
endif()
