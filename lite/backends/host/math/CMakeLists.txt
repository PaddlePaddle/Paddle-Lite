lite_cc_library(math_host SRCS
    sequence_padding.cc
    slice.cc
    pad3d.cc
    DEPS ${lite_kernel_deps})
