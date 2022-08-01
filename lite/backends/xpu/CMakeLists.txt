if(NOT LITE_WITH_XPU)
  return()
endif()

lite_cc_library(xpu_scratch SRCS xpu_scratch.cc DEPS ${xpu_builder_libs} ${xpu_runtime_libs})
lite_cc_library(xpu_quantizer SRCS xpu_quantizer.cc DEPS ${xpu_builder_libs} ${xpu_runtime_libs} xpu_scratch)
lite_cc_library(target_wrapper_xpu SRCS target_wrapper.cc DEPS ${xpu_builder_libs} ${xpu_runtime_libs} xpu_scratch xpu_quantizer)
