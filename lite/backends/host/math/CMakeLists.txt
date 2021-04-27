lite_cc_library(math_host SRCS
    sequence_padding.cc
    slice.cc
    norm.cc
    pad3d.cc
    argmax.cc
    reduce_all.cc
    concat.cc
    stack.cc
    DEPS ${lite_kernel_deps})
