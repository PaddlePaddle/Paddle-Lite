if (NOT WITH_TESTING)
    return()
endif()

if(LITE_WITH_XPU)
    lite_cc_test(test_resnet50_lite_xpu SRCS test_resnet50_lite_xpu.cc
      DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
      ${ops} ${host_kernels} ${x86_kernels} ${xpu_kernels}
      ARGS --model_dir=${LITE_MODEL_DIR}/resnet50)
    lite_cc_test(test_ernie_lite_xpu SRCS test_ernie_lite_xpu.cc
      DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
      ${ops} ${host_kernels} ${x86_kernels} ${xpu_kernels}
      ARGS --model_dir=${LITE_MODEL_DIR}/resnet50)
    lite_cc_test(test_bert_lite_xpu SRCS test_bert_lite_xpu.cc
      DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
      ${ops} ${host_kernels} ${x86_kernels} ${xpu_kernels}
      ARGS --model_dir=${LITE_MODEL_DIR}/resnet50)
endif()
