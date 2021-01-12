cc_library(math_arm_fp16 SRCS
    conv_impl_fp16.cc
    packed_sgemm_fp16.cc
    DEPS ${lite_kernel_deps} context tensor)
