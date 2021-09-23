if (LITE_ON_TINY_PUBLISH)
  return()
endif()

if(NOT WITH_COVERAGE)
    lite_cc_test(test_naive_buffer SRCS naive_buffer_test.cc DEPS model_parser)
endif()
lite_cc_test(test_naive_buffer_wrapper SRCS naive_buffer_wrapper_test.cc 
             DEPS model_parser core)
