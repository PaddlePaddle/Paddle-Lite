if(NOT LITE_WITH_XPU)
  return()
endif()

lite_cc_library(xpu_runtime SRCS runtime.cc DEPS ${xpu_runtime_libs})
lite_cc_library(xpu_builder SRCS builder.cc DEPS ${xpu_builder_libs} xpu_runtime tensor op scope)
