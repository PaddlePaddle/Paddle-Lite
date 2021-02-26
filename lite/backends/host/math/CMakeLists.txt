lite_cc_library(math_host SRCS
    sequence_padding.cc
    slice.cc
    pad3d.cc
    concat.cc
    stack.cc
    reduce_all.cc
    argmax.cc
    DEPS context)
