cc_library(math_arm_fp16 SRCS
    activation_fp16.cc
    conv_impl_fp16.cc
    conv3x3s1_depthwise_fp16.cc
    conv3x3s2_depthwise_fp16.cc
    conv5x5s1_depthwise_fp16.cc
    conv5x5s2_depthwise_fp16.cc
    conv3x3_winograd_fp16.cc
    conv3x3s1_direct_fp16.cc
    conv3x3s2_direct_fp16.cc
    elementwise_fp16.cc
    funcs_fp16.cc
    gemm_fp16.cc
    gemm_c8_fp16.cc
    gemv_fp16.cc
    interpolate_fp16.cc
    pad2d_fp16.cc
    pooling_fp16.cc
    sgemm_fp16.cc
    softmax_fp16.cc
    type_trans_fp16.cc
    shuffle_channel_fp16.cc
    DEPS ${lite_kernel_deps} context tensor)
