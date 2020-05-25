if(LITE_WITH_CV AND (NOT LITE_WITH_FPGA) AND LITE_WITH_ARM)
    lite_cc_library(paddle_cv_arm SRCS
            image_convert.cc
            bgr_rotate.cc
            paddle_image_preprocess.cc
            image2tensor.cc
            image_flip.cc
            image_rotate.cc
            image_resize.cc
            DEPS paddle_api place)
endif()
