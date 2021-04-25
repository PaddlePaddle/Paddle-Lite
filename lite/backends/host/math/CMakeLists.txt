lite_cc_library(math_host SRCS
    beam_search.cc
    sequence_padding.cc
    slice.cc
    split.cc
    gpc.cc
    norm.cc
    pad3d.cc
    concat.cc
    stack.cc
    reduce_all.cc
    argmax.cc
    topk.cc
    DEPS context)
