if (NOT LITE_WITH_PROFILE)
  return()
endif()

lite_cc_library(basic_profiler SRCS basic_profiler.cc DEPS gflags)
lite_cc_test(test_basic_profiler SRCS basic_profiler_test.cc DEPS basic_profiler)
 
lite_cc_library(lite_profiler SRCS profiler.cc DEPS context)
lite_cc_test(test_lite_timer SRCS test_timer.cc DEPS lite_profiler)
