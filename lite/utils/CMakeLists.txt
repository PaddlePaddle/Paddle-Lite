# if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
#     set(utils_DEPS)
# else()
# endif()

if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK OR LITE_ON_MODEL_OPTIMIZE_TOOL OR (NOT LITE_WITH_LOG))
  lite_cc_library(logging SRCS logging.cc)
  set(utils_DEPS logging)
  lite_cc_test(test_logging SRCS logging_test.cc DEPS ${utils_DEPS})
else()
   set(utils_DEPS glog)
endif()

lite_cc_test(test_varient SRCS varient_test.cc DEPS utils)
lite_cc_library(any SRCS any.cc)

if(LITE_ON_TINY_PUBLISH OR LITE_ON_MODEL_OPTIMIZE_TOOL)
  lite_cc_library(stream SRCS replace_stl/stream.cc)
endif()
#lite_cc_library(utils SRCS cp_logging.cc string.cc DEPS ${utils_DEPS} any)

if(LITE_ON_TINY_PUBLISH OR LITE_ON_MODEL_OPTIMIZE_TOOL)
  lite_cc_library(utils SRCS string.cc DEPS ${utils_DEPS} any stream)
else()
  lite_cc_library(utils SRCS string.cc DEPS ${utils_DEPS} any)
endif()

add_subdirectory(cv)
