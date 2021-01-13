cc_library(math_arm_fp16 SRCS
    gemm_prepacked_fp16.cc
    conv3x3s1_depthwise_fp16.cc
    conv3x3s2_depthwise_fp16.cc
    DEPS ${lite_kernel_deps} context tensor)
