if(NOT LITE_WITH_HUAWEI_ASCEND_NPU)
  return()
endif()

lite_cc_library(model_client_huawei_ascend_npu SRCS model_client.cc DEPS ${huawei_ascend_npu_runtime_libs} ${huawei_ascend_npu_builder_libs})
lite_cc_library(device_huawei_ascend_npu SRCS device.cc DEPS ${huawei_ascend_npu_runtime_libs} ${huawei_ascend_npu_builder_libs} model_client_huawei_ascend_npu)
