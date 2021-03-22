lite_cc_library(math_host SRCS
    sequence_padding.cc
    slice.cc
    gpc.cc
    pad3d.cc
    concat.cc
    stack.cc
    reduce_all.cc
    argmax.cc
    topk.cc
    DEPS context)
