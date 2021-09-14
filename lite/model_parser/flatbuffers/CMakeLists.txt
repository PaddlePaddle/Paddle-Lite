lite_cc_test(test_vector_view SRCS vector_view_test.cc DEPS model_parser core)
lite_cc_test(test_fbs_io SRCS io_test.cc DEPS model_parser core)
lite_cc_test(test_program_desc SRCS program_desc_test.cc DEPS model_parser core)
