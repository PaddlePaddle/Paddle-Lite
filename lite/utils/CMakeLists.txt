# logging system: custom logging sys will be used if 
# in tiny_publish mode or in opt tool
# otherwise, glog will be applied as logging sys
if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK OR LITE_ON_MODEL_OPTIMIZE_TOOL OR (NOT LITE_WITH_LOG))
  set(utils_deps_source
       logging.cc # Custom logging implementation
  )
else()
  set(utils_deps glog)
endif()

lite_cc_library(utils SRCS
  any.cc # any stucture
  string.cc # string structure
  replace_stl/stream.cc # Custom stream implementation
  ${utils_deps_source}
  DEPS ${utils_deps})
add_subdirectory(cv)


lite_cc_test(test_varient SRCS varient_test.cc DEPS utils)
lite_cc_test(test_utils_string SRCS string_test.cc)
# fp16
if (WITH_TESTING)
    if (LITE_WITH_CUDA)
      nv_test(float16_gpu_test SRCS float16_test.cu)
    endif ()
    lite_cc_test(float16_test SRCS float16_test.cc)
endif()
