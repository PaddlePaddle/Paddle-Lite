if(LITE_WITH_CV AND LITE_WITH_ARM)
    if(LITE_WITH_FPGA)
      lite_cc_library(paddle_cv_arm SRCS
        image_convert.cc
        bgr_rotate.cc
        paddle_image_preprocess.cc
        image2tensor.cc
        image2tensor_fpga.cc
        image_flip.cc
        image_rotate.cc
        image_resize.cc
        DEPS paddle_api place kernel_fpga)
    else()
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
endif()
