if((NOT LITE_WITH_OPENCL AND NOT LITE_WITH_FPGA) AND LITE_WITH_ARM)
    set(lite_cv_deps)
    lite_cc_library(cv_arm SRCS
            image_convert.cc
            image_preprocess.cc
            image2tensor.cc
            image_flip.cc
            image_rotate.cc
            image_transform.cc
            DEPS ${lite_cv_deps} tensor)
endif()
