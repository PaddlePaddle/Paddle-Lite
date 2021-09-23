if(LITE_WITH_CV AND (NOT LITE_WITH_OPENCL AND NOT LITE_WITH_FPGA AND NOT LITE_WITH_MLU) AND LITE_WITH_ARM)
  lite_cc_library(anakin_cv_arm SRCS
                  bgr_resize.cc
                  bgr_flip_hwc.cc
                  bgr_rotate_hwc.cc
                  bgr_to_tensor_hwc.cc
                  bgra_resize.cc
                  bgra_flip_hwc.cc
                  bgra_rotate_hwc.cc
                  bgra_to_tensor_hwc.cc
                  cv_utils.cc
                  nv12_to_bgr.cc
                  nv12_to_bgra.cc
                  nv21_to_bgr.cc
                  nv21_to_bgra.cc
                  nv21_resize.cc)
endif()
