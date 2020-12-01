function(lite_cc_test_with_model_and_data TARGET)
    if(NOT WITH_TESTING)
        return()
    endif()

    set(options "")
    set(oneValueArgs MODEL DATA CONFIG ARGS)
    set(multiValueArgs "")
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(DEFINED args_MODEL)
        set(ARGS ${ARGS} --model_dir=${LITE_MODEL_DIR}/${args_MODEL})
    endif()
    if(DEFINED args_DATA)
        set(ARGS ${ARGS} --data_dir=${LITE_MODEL_DIR}/${args_DATA})
    endif()
    if(DEFINED args_CONFIG)
        set(ARGS ${ARGS} --config_dir=${LITE_MODEL_DIR}/${args_CONFIG})
    endif()
    if(DEFINED args_ARGS)
        set(ARGS ${ARGS} ${args_ARGS})
    endif()
    lite_cc_test(${TARGET} SRCS ${TARGET}.cc
        DEPS ${lite_model_test_DEPS} paddle_api_full
        ARM_DEPS ${arm_kernels}
        X86_DEPS ${x86_kernels}
        NPU_DEPS ${npu_kernels} ${npu_bridges}
        HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels} ${huawei_ascend_npu_bridges}
        XPU_DEPS ${xpu_kernels} ${xpu_bridges}
        APU_DEPS ${apu_kernels} ${apu_bridges}
        RKNPU_DEPS ${rknpu_kernels} ${rknpu_bridges}
        BM_DEPS ${bm_kernels} ${bm_bridges}
        MLU_DEPS ${mlu_kernels} ${mlu_bridges}
        ARGS ${ARGS} SERIAL)
    if(DEFINED args_MODEL)
        add_dependencies(${TARGET} extern_lite_download_${args_MODEL}_tar_gz)
    endif()
    if(DEFINED args_DATA)
        add_dependencies(${TARGET} extern_lite_download_${args_DATA}_tar_gz)
    endif()
    if(DEFINED args_CONFIG)
        add_dependencies(${TARGET} extern_lite_download_${args_CONFIG}_tar_gz)
    endif()
endfunction()

if(LITE_WITH_ARM)
    lite_cc_test_with_model_and_data(test_inception_v4_fp32_arm MODEL inception_v4 DATA ILSVRC2012_small)
    lite_cc_test_with_model_and_data(test_mobilenet_v1_fp32_arm MODEL mobilenet_v1 DATA ILSVRC2012_small)
    lite_cc_test_with_model_and_data(test_mobilenet_v2_fp32_arm MODEL mobilenet_v2_relu DATA ILSVRC2012_small)
    lite_cc_test_with_model_and_data(test_mobilenet_v3_small_x1_0_fp32_arm MODEL mobilenet_v3_small_x1_0 DATA ILSVRC2012_small)
    lite_cc_test_with_model_and_data(test_mobilenet_v3_large_x1_0_fp32_arm MODEL mobilenet_v3_large_x1_0 DATA ILSVRC2012_small)
    lite_cc_test_with_model_and_data(test_resnet50_fp32_arm MODEL resnet50 DATA ILSVRC2012_small)
    lite_cc_test_with_model_and_data(test_squeezenet_fp32_arm MODEL squeezenet DATA ILSVRC2012_small)
    lite_cc_test_with_model_and_data(test_transformer_with_mask_fp32_arm MODEL transformer_with_mask_fp32)
endif()

if(LITE_WITH_NPU)
    lite_cc_test_with_model_and_data(test_inception_v4_fp32_huawei_kirin_npu MODEL inception_v4 DATA ILSVRC2012_small)
    lite_cc_test_with_model_and_data(test_mobilenet_v1_fp32_huawei_kirin_npu MODEL mobilenet_v1 DATA ILSVRC2012_small)
    lite_cc_test_with_model_and_data(test_mobilenet_v2_fp32_huawei_kirin_npu MODEL mobilenet_v2_relu DATA ILSVRC2012_small)
    lite_cc_test_with_model_and_data(test_mobilenet_v3_small_x1_0_fp32_huawei_kirin_npu MODEL mobilenet_v3_small_x1_0 DATA ILSVRC2012_small)
    lite_cc_test_with_model_and_data(test_mobilenet_v3_large_x1_0_fp32_huawei_kirin_npu MODEL mobilenet_v3_large_x1_0 DATA ILSVRC2012_small)
    lite_cc_test_with_model_and_data(test_resnet50_fp32_huawei_kirin_npu MODEL resnet50 DATA ILSVRC2012_small)
    lite_cc_test_with_model_and_data(test_squeezenet_fp32_huawei_kirin_npu MODEL squeezenet DATA ILSVRC2012_small)
endif()

if(LITE_WITH_XPU AND NOT LITE_WITH_XTCL)
    lite_cc_test_with_model_and_data(test_bert_fp32_baidu_xpu MODEL bert DATA bert_data)
    lite_cc_test_with_model_and_data(test_content_dnn_fp32_baidu_xpu MODEL content_dnn_model DATA content_dnn_data)
    lite_cc_test_with_model_and_data(test_ernie_fp32_baidu_xpu MODEL ernie DATA bert_data)
    lite_cc_test_with_model_and_data(test_googlenet_fp32_baidu_xpu MODEL GoogLeNet DATA ILSVRC2012_small)
    lite_cc_test_with_model_and_data(test_mmdnn_fp32_baidu_xpu MODEL mmdnn_model DATA mmdnn_data)
    lite_cc_test_with_model_and_data(test_resnet50_fp32_baidu_xpu MODEL resnet50_model DATA ILSVRC2012_small)
    lite_cc_test_with_model_and_data(test_sfa_fp32_baidu_xpu MODEL sfa_model DATA sfa_data)
    lite_cc_test_with_model_and_data(test_vgg19_fp32_baidu_xpu MODEL VGG19 DATA ILSVRC2012_small)
endif()

if(LITE_WITH_RKNPU)
    lite_cc_test_with_model_and_data(test_mobilenet_v1_int8_rockchip_npu MODEL mobilenet_v1_int8_for_rockchip_npu DATA ILSVRC2012_small)
endif()

if(LITE_WITH_APU)
    lite_cc_test_with_model_and_data(test_mobilenet_v1_int8_mediatek_apu MODEL mobilenet_v1_int8_for_mediatek_apu DATA ILSVRC2012_small)
endif()

if(LITE_WITH_IMAGINATION_NNA)
    lite_cc_test_with_model_and_data(test_mobilenet_v1_int8_imagination_nna MODEL mobilenet_v1_int8_for_imagination_nna DATA ILSVRC2012_small)
endif()
