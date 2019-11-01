if(NOT LITE_WITH_NPU)
  return()
endif()

lite_cc_library(npu_runtime SRCS runtime.cc DEPS ${npu_runtime_libs})
lite_cc_library(npu_builder SRCS builder.cc DEPS ${npu_builder_libs} npu_runtime tensor op scope)
