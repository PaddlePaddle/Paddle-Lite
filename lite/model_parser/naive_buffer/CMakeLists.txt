lite_cc_library(naive_buffer SRCS naive_buffer.cc DEPS types_lite)
lite_cc_test(test_naive_buffer SRCS naive_buffer_test.cc DEPS naive_buffer)

lite_cc_library(nb_op_desc SRCS op_desc.cc DEPS naive_buffer)
