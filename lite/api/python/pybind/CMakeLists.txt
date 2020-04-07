set(PYBIND_DEPS pybind python paddle_api_light paddle_api)
if (NOT LITE_ON_TINY_PUBLISH)
   set(PYBIND_DEPS ${PYBIND_DEPS} paddle_api_full opt_base)
endif()

lite_cc_library(lite_pybind SHARED SRCS pybind.cc DEPS ${PYBIND_DEPS})
if (LITE_ON_TINY_PUBLISH)
   set_target_properties(lite_pybind PROPERTIES COMPILE_FLAGS "-flto -fdata-sections")
endif()
