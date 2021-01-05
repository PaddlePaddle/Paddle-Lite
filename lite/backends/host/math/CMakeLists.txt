lite_cc_library(math_host SRCS
    sequence_padding.cc
    slice.cc
    concat.cc
    stack.cc
    reduce_all.cc
    DEPS context)
