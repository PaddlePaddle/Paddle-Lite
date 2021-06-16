lite_cc_library(math_host SRCS
    beam_search.cc
    sequence_padding.cc
    slice.cc
    split.cc
    gpc.cc
    norm.cc
    pad3d.cc
    prior_box.cc
    concat.cc
    stack.cc
    reduce.cc
    argmax.cc
    topk.cc
    yolo_box.cc
    DEPS context)
