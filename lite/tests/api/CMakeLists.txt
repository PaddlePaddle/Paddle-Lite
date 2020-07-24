if(LITE_WITH_ARM)
    lite_cc_test(test_transformer_with_mask_fp32_arm SRCS test_transformer_with_mask_fp32_arm.cc
      DEPS ${lite_model_test_DEPS} paddle_api_full
      ARM_DEPS ${arm_kernels}
      ARGS --model_dir=${LITE_MODEL_DIR}/transformer_with_mask_fp32 SERIAL)
  if(WITH_TESTING)
      add_dependencies(test_transformer_with_mask_fp32_arm extern_lite_download_transformer_with_mask_fp32_tar_gz)
  endif()
endif()

if(LITE_WITH_XPU)
    lite_cc_test(test_resnet50_lite_xpu SRCS test_resnet50_lite_xpu.cc
      DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
      ${ops} ${host_kernels} ${x86_kernels} ${xpu_kernels}
      ARGS --model_dir=${LITE_MODEL_DIR}/resnet50)
    lite_cc_test(test_ernie_lite_xpu SRCS test_ernie_lite_xpu.cc
      DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
      ${ops} ${host_kernels} ${x86_kernels} ${xpu_kernels}
      ARGS --model_dir=${LITE_MODEL_DIR}/ernie)
    lite_cc_test(test_bert_lite_xpu SRCS test_bert_lite_xpu.cc
      DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
      ${ops} ${host_kernels} ${x86_kernels} ${xpu_kernels}
      ARGS --model_dir=${LITE_MODEL_DIR}/bert)
    if(WITH_TESTING)
        add_dependencies(test_resnet50_lite_xpu extern_lite_download_resnet50_tar_gz)
        add_dependencies(test_ernie_lite_xpu extern_lite_download_ernie_tar_gz)
        add_dependencies(test_bert_lite_xpu extern_lite_download_bert_tar_gz)
    endif()
    # TODO(miaotianxiang): enable later
    #lite_cc_test(test_fpr_lite_xpu SRCS test_fpr_lite_xpu.cc
      #DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
      #${ops} ${host_kernels} ${x86_kernels} ${xpu_kernels}
      #ARGS --model_dir=${LITE_MODEL_DIR}/resnet50)
    #lite_cc_test(test_mmdnn_lite_xpu SRCS test_mmdnn_lite_xpu.cc
      #DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
      #${ops} ${host_kernels} ${x86_kernels} ${xpu_kernels}
      #ARGS --model_dir=${LITE_MODEL_DIR}/resnet50)
endif()

if(LITE_WITH_RKNPU)
    lite_cc_test(test_mobilenetv1_int8_rknpu SRCS test_mobilenetv1_int8_rknpu.cc
      DEPS ${lite_model_test_DEPS} paddle_api_full
      RKNPU_DEPS ${rknpu_kernels} ${rknpu_bridges}
      ARGS --model_dir=${LITE_MODEL_DIR}/MobilenetV1_full_quant SERIAL)
endif()

if(LITE_WITH_APU)
    lite_cc_test(test_mobilenetv1_int8_apu SRCS test_mobilenetv1_int8_apu.cc
      DEPS ${lite_model_test_DEPS} paddle_api_full
      APU_DEPS ${apu_kernels} ${apu_bridges}
      ARGS --model_dir=${LITE_MODEL_DIR}/MobilenetV1_full_quant SERIAL)
endif()
