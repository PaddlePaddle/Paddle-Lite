if (NOT LITE_ON_TINY_PUBLISH)
  lite_cc_test(test_ssa_program SRCS ssa_program_test.cc DEPS model_parser)
endif()
