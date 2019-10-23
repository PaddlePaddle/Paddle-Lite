set(PYBIND_DEPS pybind python paddle_api_light paddle_api)
if (NOT LITE_ON_TINY_PUBLISH)
   set(PYBIND_DEPS ${PYBIND_DEPS} paddle_api_full)
endif()

lite_cc_library(lite_pybind SHARED SRCS pybind.cc DEPS ${PYBIND_DEPS})
