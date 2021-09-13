# To make sure the test framework is only actived in TESTING mode.
if(NOT WITH_TESTING)
    return()
endif()


if((NOT LITE_WITH_OPENCL) AND (LITE_WITH_X86 OR LITE_WITH_ARM))
  lite_cc_test(test_arena_framework SRCS framework_test.cc DEPS core ops kernels)
endif()
