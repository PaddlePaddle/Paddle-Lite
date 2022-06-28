include(lite)

message(STATUS "LITE_WITH_CUDA:\t${LITE_WITH_CUDA}")
message(STATUS "LITE_WITH_X86:\t${LITE_WITH_X86}")
message(STATUS "LITE_WITH_ARM:\t${LITE_WITH_ARM}")
message(STATUS "LITE_WITH_SW:\t${LITE_WITH_SW}")
message(STATUS "LITE_WITH_OPENCL:\t${LITE_WITH_OPENCL}")
message(STATUS "LITE_WITH_METAL:\t${LITE_WITH_METAL}")
message(STATUS "LITE_WITH_NPU:\t${LITE_WITH_NPU}")
message(STATUS "LITE_WITH_XPU:\t${LITE_WITH_XPU}")
message(STATUS "LITE_WITH_FPGA:\t${LITE_WITH_FPGA}")
message(STATUS "LITE_WITH_INTEL_FPGA:\t${LITE_WITH_INTEL_FPGA}")
message(STATUS "LITE_WITH_MLU:\t${LITE_WITH_MLU}")
message(STATUS "LITE_WITH_BM:\t${LITE_WITH_BM}")
message(STATUS "LITE_WITH_NNADAPTER:\t${LITE_WITH_NNADAPTER}")
message(STATUS "LITE_WITH_PROFILE:\t${LITE_WITH_PROFILE}")
message(STATUS "LITE_WITH_CV:\t${LITE_WITH_CV}")

set(LITE_MODEL_DIR "${THIRD_PARTY_PATH}/install")
set(LITE_ON_MOBILE ${LITE_WITH_ARM})

add_subdirectory(backends)
add_subdirectory(utils)
add_subdirectory(operators)
add_subdirectory(kernels)
add_subdirectory(core)
add_subdirectory(model_parser)
add_subdirectory(api)

if (NOT LITE_ON_TINY_PUBLISH)
    add_subdirectory(tests)
    add_subdirectory(tools)
endif()

if (WITH_TESTING)
    set(LITE_URL_FOR_UNITTESTS "http://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests")
    # models
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "lite_naive_model.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "mobilenet_v1.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "mobilenet_v2_relu.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "inception_v4_simple.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "GoogleNet_inference.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "inception_v4.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "squeezenet.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "mobilenet_v3_small_x1_0.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "mobilenet_v3_large_x1_0.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "mobilenet_v1_int8_per_layer.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "mobilenet_v1_int8_per_channel.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "resnet50_int8_per_layer.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ssd_mobilenet_v1_relu_voc_fp32_300.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ssd_mobilenet_v1_relu_voc_int8_300_per_layer.tar.gz")
    if (LITE_WITH_NNADAPTER)
        set(LITE_URL_FOR_NNADAPTER_UNITTESTS "http://paddlelite-demo.bj.bcebos.com/NNAdapter/models")
        # PaddleClas
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "AlexNet_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "DenseNet121_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "GoogLeNet_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "EfficientNetB0_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "InceptionV3_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "InceptionV4_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "MobileNetV1_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "MobileNetV2_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "MobileNetV3_large_x1_0_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "MobileNetV3_small_x1_0_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ResNet18_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ResNet50_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ResNet101_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ResNeXt50_32x4d_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ShuffleNetV2_x1_0_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "SqueezeNet1_0_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "VGG16_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "VGG19_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "DarkNet53.tar.gz"             MODEL_PATH "PaddleClas/v2.3/DarkNet")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "DeiT_base_patch16_224.tar.gz" MODEL_PATH "PaddleClas/v2.3/DeiT")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "DPN68.tar.gz"                 MODEL_PATH "PaddleClas/v2.3/DPN")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "GhostNet_x1_0.tar.gz"         MODEL_PATH "PaddleClas/v2.3/GhostNet")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "HRNet_W18_C.tar.gz"           MODEL_PATH "PaddleClas/v2.3/HRNet")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "PPLCNet_x0_25.tar.gz"         MODEL_PATH "PaddleClas/v2.3/PPLCNet")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "Res2Net50_26w_4s.tar.gz"      MODEL_PATH "PaddleClas/v2.3/Res2Net")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "SE_ResNet50_vd.tar.gz"        MODEL_PATH "PaddleClas/v2.3/SENet")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "ViT_base_patch16_224.tar.gz"  MODEL_PATH "PaddleClas/v2.3/VisionTransformer")
        # PaddleDetection
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ssd_mobilenet_v1_relu_voc_v2_2.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ssd_vgg16_voc_v2_2.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "yolov3_darknet53_coco_v2_2.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "yolov3_mobilenet_v1_coco_v2_2.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "yolov3_mobilenet_v3_large_coco_v2_2.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "yolov3_r50vd_dcn_coco_v2_2.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "yolov4_cspdarknet_coco_v1_8.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ppyolo_r50vd_dcn_1x_coco_v2_2.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "higherhrnet_hrnet_w32_640_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "hrnet_w32_384x288_v2_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "picodet_m_416_coco.tar.gz"           MODEL_PATH "PaddleDetection/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "ppyolo_mbv3_large_coco.tar.gz"       MODEL_PATH "PaddleDetection/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "ppyolo_r50vd_dcn_1x_coco.tar.gz"     MODEL_PATH "PaddleDetection/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "ppyolo_tiny_650e_coco.tar.gz"        MODEL_PATH "PaddleDetection/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "ppyolov2_r50vd_dcn_365e_coco.tar.gz" MODEL_PATH "PaddleDetection/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "tinypose_128x96.tar.gz"              MODEL_PATH "PaddleDetection/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "ssdlite_mobilenet_v3_small.tar.gz"   MODEL_PATH "PaddleDetection/static")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "ssdlite_mobilenet_v3_large.tar.gz"   MODEL_PATH "PaddleDetection/static")
        # PaddleSeg
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "bisenet.tar.gz"   MODEL_PATH "PaddleSeg/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "segformer.tar.gz" MODEL_PATH "PaddleSeg/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "stdcseg.tar.gz"   MODEL_PATH "PaddleSeg/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "unet_cityscapes_1024x512_160k.tar.gz"                   MODEL_PATH "PaddleSeg/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.tar.gz" MODEL_PATH "PaddleSeg/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "pphumanseg_lite_generic_192x192_with_softmax.tar.gz"    MODEL_PATH "PaddleSeg/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax.tar.gz" MODEL_PATH "PaddleSeg/v2.3")
        # PaddleOCR
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "rec_crnn_mv3_ctc.tar.gz"               MODEL_PATH "PaddleOCR")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "ch_ppocr_mobile_v2_0_rec_v2_0.tar.gz"  MODEL_PATH "PaddleOCR/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "ch_ppocr_mobile_v2_0_det_v2_0.tar.gz"  MODEL_PATH "PaddleOCR/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "ch_ppocr_server_v2.0_rec_infer.tar.gz" MODEL_PATH "PaddleOCR/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "ch_ppocr_server_v2.0_det_infer.tar.gz" MODEL_PATH "PaddleOCR/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "ch_PP-OCRv2_det_infer.tar.gz"          MODEL_PATH "PaddleOCR/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "ch_PP-OCRv2_rec_infer.tar.gz"          MODEL_PATH "PaddleOCR/v2.3")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "e2e_server_pgnetA.tar.gz"              MODEL_PATH "PaddleOCR/v2.3")
        # PaddleNLP
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "ernie_tiny.tar.gz"           MODEL_PATH "PaddleNLP")
        # PaddleGAN
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "esrgan_psnr_x4_div2k.tar.gz" MODEL_PATH "PaddleGAN")
        # PaddleRec
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "deepfm.tar.gz"    MODEL_PATH "PaddleRec/v2.1.0")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "naml.tar.gz"      MODEL_PATH "PaddleRec/v2.1.0")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "ncf.tar.gz"       MODEL_PATH "PaddleRec/v2.1.0")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "wide_deep.tar.gz" MODEL_PATH "PaddleRec/v2.1.0")
        # PaddleVideo
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_NNADAPTER_UNITTESTS} "ppTSN.tar.gz"     MODEL_PATH "PaddleVideo/v2.2.0")
    endif()
    if(LITE_WITH_ARM)
	    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "mobilenet_v1_int16.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "resnet50.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "MobileNetV1_quant.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "transformer_with_mask_fp32.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "mobilenet_v1_int8_for_arm.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "mobilenetv1_int8_dygraph_for_arm.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "mobilenet_v2_int8_for_arm.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "resnet50_int8_for_arm.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "mobilenet_v1_int8_for_mediatek_apu.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "mobilenet_v1_int8_for_rockchip_npu.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "mobilenet_v1_int8_for_imagination_nna.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "fast_rcnn_fluid184.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ocr_rec_quant_mul_lstm_for_arm.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "nlp_quant_lstm_int8_arm.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "lac_fp32_arm.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "transformer_nlp2_fp32_arm.tar.gz")
    else()
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "step_rnn.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "resnet50.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "resnet50_vd.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "bert.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "bert_base_chinese.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ernie.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "GoogLeNet.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "VGG19.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "yolov3_darknet53.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "vgg16.tar.gz")
    endif()
    # data
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ILSVRC2012_500.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "flowers102_val.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "bert_data.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "bert_base_chinese_data.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ocr_rec_img_txt.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "nlp_quant_lstm_int8_data_txt.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "lac_data_txt.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "transformer_nlp2_data_txt.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "roadsign_data_128.tar.gz")
    if (LITE_WITH_NNADAPTER)
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "PASCALVOC2012_500.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "VOC2012_100.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "COCO2017_100.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ocr_db_data.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ocr_rec_data.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ocr_e2e_data.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ernie_tiny_data.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "COCO2017_keypoints_100.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "GAN_set14.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "specific_model_input_output.tar.gz")
    endif()
endif()

# ----------------------------- PUBLISH -----------------------------
# The final target for publish lite lib
add_custom_target(publish_inference)
if (LITE_WITH_ARM)
    # for publish
    set(INFER_LITE_PUBLISH_ROOT "${PADDLE_BINARY_DIR}/inference_lite_lib.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}")
    if (LITE_WITH_OPENCL)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.opencl")
    endif(LITE_WITH_OPENCL)
    if (LITE_WITH_METAL)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.metal")
    endif(LITE_WITH_METAL)
    if (LITE_WITH_NPU)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.npu")
    endif(LITE_WITH_NPU)
    if (LITE_WITH_XPU)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.xpu")
    endif(LITE_WITH_XPU)
    if (LITE_WITH_FPGA)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.fpga")
    endif(LITE_WITH_FPGA)
    if (LITE_WITH_BM)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.bm")
    endif(LITE_WITH_BM)
    if (LITE_WITH_INTEL_FPGA)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.intel_fpga")
    endif(LITE_WITH_INTEL_FPGA)
    if (LITE_WITH_NNADAPTER)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.nnadapter")
    endif(LITE_WITH_NNADAPTER)
else()
    set(INFER_LITE_PUBLISH_ROOT "${PADDLE_BINARY_DIR}/inference_lite_lib")
endif()
message(STATUS "publish inference lib to ${INFER_LITE_PUBLISH_ROOT}")

# add python lib
if (LITE_WITH_PYTHON)
    if(WIN32)
        set(LITE_CORE "${PADDLE_BINARY_DIR}/lite/api/${CMAKE_BUILD_TYPE}/lite.pyd")
        set(LITE_CORE_DEPS ${LITE_CORE})
        add_custom_command(OUTPUT   ${LITE_CORE}
            COMMAND cmake -E copy $<TARGET_FILE:lite_pybind> ${LITE_CORE}
            DEPENDS lite_pybind)
        add_custom_target(copy_lite_pybind ALL DEPENDS ${LITE_CORE_DEPS})

        add_custom_target(publish_inference_python_lib ${TARGET}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/python/lib"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/python/install/libs"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/python/install/lite"
            COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_BINARY_DIR}/lite/api/python/setup.py" "${INFER_LITE_PUBLISH_ROOT}/python/install/"
            COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_SOURCE_DIR}/lite/api/python/__init__.py" "${INFER_LITE_PUBLISH_ROOT}/python/install/lite"
            COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_BINARY_DIR}/lite/api/${CMAKE_BUILD_TYPE}/lite.pyd" "${INFER_LITE_PUBLISH_ROOT}/python/install/lite/lite.pyd"
            COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_BINARY_DIR}/lite/api/${CMAKE_BUILD_TYPE}/lite.pyd" "${INFER_LITE_PUBLISH_ROOT}/python/lib/lite.pyd"
            DEPENDS copy_lite_pybind
            )

        add_custom_target(publish_inference_python_installer ${TARGET}
            COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_wheel
            WORKING_DIRECTORY ${INFER_LITE_PUBLISH_ROOT}/python/install/
            DEPENDS  publish_inference_python_lib)
        add_custom_target(publish_inference_python_light_demo ${TARGET}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/demo/python"
            COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_SOURCE_DIR}/lite/demo/python/mobilenetv1_light_api.py" "${INFER_LITE_PUBLISH_ROOT}/demo/python/"
            COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_SOURCE_DIR}/lite/demo/python/mobilenetv1_full_api.py" "${INFER_LITE_PUBLISH_ROOT}/demo/python/"
            )
        add_dependencies(publish_inference publish_inference_python_lib)
        add_dependencies(publish_inference publish_inference_python_installer)
        add_dependencies(publish_inference publish_inference_python_light_demo)
    else()
    if(APPLE)
        add_custom_target(publish_inference_python_lib ${TARGET}
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/python/lib"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/python/install/libs"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/python/install/lite"
                COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/python/setup.py" "${INFER_LITE_PUBLISH_ROOT}/python/install/"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/api/python/__init__.py" "${INFER_LITE_PUBLISH_ROOT}/python/install/lite"
                COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/liblite_pybind.dylib" "${INFER_LITE_PUBLISH_ROOT}/python/install/lite/lite.so"
                COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/liblite_pybind.dylib" "${INFER_LITE_PUBLISH_ROOT}/python/lib/lite.so")
    else()
        add_custom_target(publish_inference_python_lib ${TARGET}
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/python/lib"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/python/install/libs"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/python/install/lite"
                COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/python/setup.py" "${INFER_LITE_PUBLISH_ROOT}/python/install/"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/api/python/__init__.py" "${INFER_LITE_PUBLISH_ROOT}/python/install/lite"
                COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/liblite_pybind.so" "${INFER_LITE_PUBLISH_ROOT}/python/install/lite/lite.so"
                COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/liblite_pybind.so" "${INFER_LITE_PUBLISH_ROOT}/python/lib/lite.so")
    endif()
    add_custom_target(publish_inference_python_installer ${TARGET}
        COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_wheel
        WORKING_DIRECTORY ${INFER_LITE_PUBLISH_ROOT}/python/install/
        DEPENDS publish_inference_python_lib)
    add_custom_target(publish_inference_python_light_demo ${TARGET}
    	COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/python"
    	COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/python/mobilenetv1_light_api.py" "${INFER_LITE_PUBLISH_ROOT}/demo/python/")
    if (NOT LITE_ON_TINY_PUBLISH)
        add_custom_target(publish_inference_python_full_demo ${TARGET}
    	    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/python"
    	    COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/python/mobilenetv1_full_api.py" "${INFER_LITE_PUBLISH_ROOT}/demo/python/")
        add_dependencies(publish_inference publish_inference_python_full_demo)
    endif()
    add_dependencies(publish_inference_python_lib lite_pybind)
    add_dependencies(publish_inference publish_inference_python_lib)
    add_dependencies(publish_inference publish_inference_python_installer)
    add_dependencies(publish_inference publish_inference_python_light_demo)
    endif(WIN32)
endif()

if (LITE_WITH_CUDA OR LITE_WITH_X86)
    if(APPLE)
        add_custom_target(publish_inference_cxx_lib ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/bin"
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND cp "${PADDLE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/paddle_use_kernels.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/paddle_use_ops.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND cp "${PADDLE_BINARY_DIR}/libpaddle_api_full_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND cp "${PADDLE_BINARY_DIR}/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/*.dylib" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            )
        add_custom_target(publish_inference_third_party ${TARGET}
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/third_party"
                COMMAND cp -r "${PADDLE_BINARY_DIR}/third_party/install/*" "${INFER_LITE_PUBLISH_ROOT}/third_party")
        add_dependencies(publish_inference_cxx_lib bundle_full_api)
        add_dependencies(publish_inference_cxx_lib bundle_light_api)
        add_dependencies(publish_inference_cxx_lib paddle_full_api_shared)
        add_dependencies(publish_inference_cxx_lib paddle_light_api_shared)
        add_dependencies(publish_inference publish_inference_cxx_lib)
        add_dependencies(publish_inference publish_inference_third_party)
    elseif(NOT WIN32)
        add_custom_target(publish_inference_cxx_lib ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/bin"
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND cp "${PADDLE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/paddle_use_kernels.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/paddle_use_ops.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND cp "${PADDLE_BINARY_DIR}/libpaddle_api_full_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND cp "${PADDLE_BINARY_DIR}/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/*.so" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            )
        if (LITE_WITH_CUDA)
            add_custom_target(publish_inference_third_party ${TARGET}
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/third_party"
                    COMMAND cp -r "${PADDLE_BINARY_DIR}/third_party/install/*" "${INFER_LITE_PUBLISH_ROOT}/third_party")
            add_dependencies(publish_inference publish_inference_third_party)
        endif()
        add_dependencies(publish_inference_cxx_lib bundle_full_api)
        add_dependencies(publish_inference_cxx_lib bundle_light_api)
        add_dependencies(publish_inference_cxx_lib paddle_full_api_shared)
        add_dependencies(publish_inference_cxx_lib paddle_light_api_shared)
        add_dependencies(publish_inference publish_inference_cxx_lib)
    endif()
endif()

if (LITE_WITH_X86)
  if(WIN32)
        if(${CMAKE_GENERATOR}  MATCHES "Ninja")
            add_custom_target(publish_inference_x86_cxx_lib ${TARGET}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/bin"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_BINARY_DIR}/lite/api/test_model_bin.exe" "${INFER_LITE_PUBLISH_ROOT}/bin"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_SOURCE_DIR}/lite/api/paddle_api.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_SOURCE_DIR}/lite/api/paddle_place.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_BINARY_DIR}/lite/api/paddle_use_kernels.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_BINARY_DIR}/lite/api/paddle_use_ops.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_SOURCE_DIR}/lite/api/paddle_use_passes.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_SOURCE_DIR}/lite/api/paddle_lite_factory_helper.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_BINARY_DIR}/lite/api/libpaddle_api_full_bundled.lib" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_BINARY_DIR}/lite/api/libpaddle_api_light_bundled.lib" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            )
        else()
            add_custom_target(publish_inference_x86_cxx_lib ${TARGET}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/bin"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_BINARY_DIR}/lite/api//${CMAKE_BUILD_TYPE}/test_model_bin.exe" "${INFER_LITE_PUBLISH_ROOT}/bin"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_SOURCE_DIR}/lite/api/paddle_api.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_SOURCE_DIR}/lite/api/paddle_place.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_BINARY_DIR}/lite/api/paddle_use_kernels.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_BINARY_DIR}/lite/api/paddle_use_ops.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_SOURCE_DIR}/lite/api/paddle_use_passes.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_SOURCE_DIR}/lite/api/paddle_lite_factory_helper.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_BINARY_DIR}/lite/api/${CMAKE_BUILD_TYPE}/libpaddle_api_full_bundled.lib" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_BINARY_DIR}/lite/api/${CMAKE_BUILD_TYPE}/libpaddle_api_light_bundled.lib" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            )
        endif()

        add_dependencies(publish_inference_x86_cxx_lib test_model_bin)
        add_dependencies(publish_inference_x86_cxx_lib bundle_full_api)
        add_dependencies(publish_inference_x86_cxx_lib bundle_light_api)
        add_dependencies(publish_inference publish_inference_x86_cxx_lib)

        configure_file(${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt.in ${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt @ONLY)
        configure_file(${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt.in ${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt @ONLY)
        configure_file(${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/build.bat.in ${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/build.bat @ONLY)
        configure_file(${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/build.bat.in ${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/build.bat @ONLY)

        add_custom_target(publish_inference_x86_cxx_demos ${TARGET}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/third_party/mklml"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${PADDLE_BINARY_DIR}/third_party/install/mklml" "${INFER_LITE_PUBLISH_ROOT}/third_party/mklml"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full"
            COMMAND ${CMAKE_COMMAND} -E remove "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light/CMakeLists.txt.in"
            COMMAND ${CMAKE_COMMAND} -E remove "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full/CMakeLists.txt.in"
            COMMAND ${CMAKE_COMMAND} -E remove "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light/build.bat.in"
            COMMAND ${CMAKE_COMMAND} -E remove "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full/build.bat.in"
            COMMAND ${CMAKE_COMMAND} -E remove "${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt"
            COMMAND ${CMAKE_COMMAND} -E remove "${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt"
        )
        add_dependencies(publish_inference_x86_cxx_lib publish_inference_x86_cxx_demos)
        add_dependencies(publish_inference_x86_cxx_demos paddle_api_full_bundled eigen3)

  else()
    add_custom_target(publish_inference_x86_cxx_lib ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/bin"
            COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/test_model_bin" "${INFER_LITE_PUBLISH_ROOT}/bin"
            )
    add_dependencies(publish_inference_x86_cxx_lib test_model_bin)

    configure_file(${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt.in ${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt @ONLY)
    configure_file(${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt.in ${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt @ONLY)
    add_custom_target(publish_inference_x86_cxx_demos ${TARGET}
           COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
           COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light"
           COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full"
           COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/third_party"
           COMMAND rm -rf "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light/*.in"
           COMMAND rm -rf "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full/*.in"
           COMMAND rm -rf "${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt"
           COMMAND rm -rf "${PADDLE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt"
       )
    if(WITH_MKL)
        add_custom_command(TARGET publish_inference_x86_cxx_demos POST_BUILD
            COMMAND cp -r "${PADDLE_BINARY_DIR}/third_party/install/mklml" "${INFER_LITE_PUBLISH_ROOT}/third_party/")
    endif()
    add_dependencies(publish_inference_x86_cxx_lib publish_inference_x86_cxx_demos)
    add_dependencies(publish_inference_x86_cxx_demos paddle_full_api_shared eigen3)
    add_dependencies(publish_inference publish_inference_x86_cxx_lib)
    add_dependencies(publish_inference publish_inference_x86_cxx_demos)
  endif()
endif()

if(LITE_WITH_CUDA)
    add_custom_target(publish_inference_cuda_cxx_demos ${TARGET}
           COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
           COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/cuda_demo/*" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
           )
    add_dependencies(publish_inference_cuda_cxx_demos paddle_full_api_shared)
    add_dependencies(publish_inference publish_inference_cuda_cxx_demos)
endif(LITE_WITH_CUDA)

if (LITE_WITH_ARM)
    if (NOT LITE_ON_TINY_PUBLISH)
        # add cxx lib
        add_custom_target(publish_inference_cxx_lib ${TARGET}
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/bin"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/paddle_use_kernels.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/paddle_use_ops.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND cp "${PADDLE_BINARY_DIR}/libpaddle_api_full_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND cp "${PADDLE_BINARY_DIR}/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/test_model_bin" "${INFER_LITE_PUBLISH_ROOT}/bin"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/utils/cv/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                )
            if(NOT IOS)
                add_dependencies(publish_inference_cxx_lib bundle_full_api)
                add_dependencies(publish_inference_cxx_lib bundle_light_api)
                add_dependencies(publish_inference_cxx_lib test_model_bin)
                add_dependencies(publish_inference_cxx_lib benchmark_bin)
                if (ARM_TARGET_OS STREQUAL "android" OR ARM_TARGET_OS STREQUAL "armlinux" OR ARM_TARGET_OS STREQUAL "armmacos" OR ARM_TARGET_OS STREQUAL "qnx")
                    add_dependencies(publish_inference_cxx_lib paddle_full_api_shared)
                    add_dependencies(publish_inference paddle_light_api_shared)
                    add_custom_command(TARGET publish_inference_cxx_lib
                          COMMAND cp ${PADDLE_BINARY_DIR}/lite/api/libpaddle_light_api_shared* ${INFER_LITE_PUBLISH_ROOT}/cxx/lib
                          COMMAND cp ${PADDLE_BINARY_DIR}/lite/api/libpaddle_full_api_shared* ${INFER_LITE_PUBLISH_ROOT}/cxx/lib
                          COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/tools/benchmark/benchmark_bin" "${INFER_LITE_PUBLISH_ROOT}/bin"
                          )
                elseif(ARM_TARGET_OS STREQUAL "armmacos")
                    add_dependencies(publish_inference_cxx_lib paddle_full_api_shared)
                    add_dependencies(publish_inference paddle_light_api_shared)
                    add_custom_command(TARGET publish_inference_cxx_lib
                        COMMAND cp ${PADDLE_BINARY_DIR}/lite/api/*.dylib ${INFER_LITE_PUBLISH_ROOT}/cxx/lib
                        COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/tools/benchmark/benchmark_bin" "${INFER_LITE_PUBLISH_ROOT}/bin"
                        )
                endif()
                add_dependencies(publish_inference publish_inference_cxx_lib)
            endif()
    else()
        if (IOS)
            if(${CMAKE_GENERATOR} STREQUAL "Xcode")
                set(IOS_BUILD_DIR "$(CONFIGURATION)$(EFFECTIVE_PLATFORM_NAME)")
            endif()
            add_custom_target(tiny_publish_cxx_lib ${TARGET}
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/lib"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/include"
                    COMMAND cp "${PADDLE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/include"
                    COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/paddle_use_kernels.h" "${INFER_LITE_PUBLISH_ROOT}/include"
                    COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/paddle_use_ops.h" "${INFER_LITE_PUBLISH_ROOT}/include"
                    COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/${IOS_BUILD_DIR}/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/lib"
                    COMMAND cp "${PADDLE_SOURCE_DIR}/lite/utils/cv/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/include"
                    )
            add_dependencies(tiny_publish_cxx_lib paddle_api_light_bundled)
            add_dependencies(publish_inference tiny_publish_cxx_lib)
        else()
            if ((ARM_TARGET_OS STREQUAL "android") OR (ARM_TARGET_OS STREQUAL "armlinux") OR (ARM_TARGET_OS STREQUAL "qnx"))
                # compile cplus shared library, pack the cplus demo and lib into the publish directory.
                add_custom_target(tiny_publish_cxx_lib ${TARGET}
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                    COMMAND cp "${PADDLE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                    COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/paddle_use_kernels.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                    COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/paddle_use_ops.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                    COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/libpaddle_light_api_shared.so" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                    COMMAND cp "${PADDLE_SOURCE_DIR}/lite/utils/cv/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                    )
                add_dependencies(tiny_publish_cxx_lib paddle_light_api_shared)
                add_dependencies(publish_inference tiny_publish_cxx_lib)
                if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
                    add_custom_command(TARGET tiny_publish_cxx_lib POST_BUILD
                                COMMAND ${CMAKE_STRIP} "-s" ${INFER_LITE_PUBLISH_ROOT}/cxx/lib/libpaddle_light_api_shared.so)
                endif()
                # compile cplus static library, pack static lib into the publish directory.
                if(LITE_WITH_STATIC_LIB)
                    add_custom_target(tiny_publish_cxx_static_lib ${TARGET}
                        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx"
                        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                        COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                        )
                    add_dependencies(tiny_publish_cxx_static_lib paddle_api_light_bundled)
                    add_dependencies(publish_inference tiny_publish_cxx_static_lib)
                endif()
            elseif(ARM_TARGET_OS STREQUAL "armmacos")
                add_custom_target(tiny_publish_cxx_lib ${TARGET}
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                    COMMAND cp "${PADDLE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                    COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/paddle_use_kernels.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                    COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/paddle_use_ops.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                    COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/*.dylib" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                    COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                    COMMAND cp "${PADDLE_SOURCE_DIR}/lite/utils/cv/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                    COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/armmacos_mobile_light_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                    COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/mobile_light/*" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/armmacos_mobile_light_demo"
                )
                add_dependencies(tiny_publish_cxx_lib paddle_api_light_bundled)
                add_dependencies(tiny_publish_cxx_lib paddle_light_api_shared)
                add_dependencies(publish_inference tiny_publish_cxx_lib)
            endif()
        endif()
    endif()


    if (LITE_WITH_JAVA)
        # add java lib
        add_custom_target(publish_inference_java_lib ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/java/so"
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/java/jar"
            COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/android/jni/native/libpaddle_lite_jni.so" "${INFER_LITE_PUBLISH_ROOT}/java/so"
            COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/android/jni/PaddlePredictor.jar" "${INFER_LITE_PUBLISH_ROOT}/java/jar"
            COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/api/android/jni/src" "${INFER_LITE_PUBLISH_ROOT}/java"
        )
        add_dependencies(publish_inference_java_lib paddle_lite_jni PaddlePredictor)
        add_dependencies(publish_inference publish_inference_java_lib)
        if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            add_custom_command(TARGET publish_inference_java_lib POST_BUILD
                                       COMMAND ${CMAKE_STRIP} "-s" ${INFER_LITE_PUBLISH_ROOT}/java/so/libpaddle_lite_jni.so)
        endif()
    endif()

    if ((ARM_TARGET_OS STREQUAL "android") AND
            ((ARM_TARGET_ARCH_ABI STREQUAL armv7) OR (ARM_TARGET_ARCH_ABI STREQUAL armv8)))
        if (NOT LITE_ON_TINY_PUBLISH)
            # copy
            add_custom_target(publish_inference_android_cxx_demos ${TARGET}
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/third_party"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/include"
                COMMAND cp -r "${PADDLE_BINARY_DIR}/third_party/install/gflags" "${INFER_LITE_PUBLISH_ROOT}/third_party"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/Makefile.def" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/README.md" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/mobile_full" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/mobile_full/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobile_full/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/mobile_light" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/mobile_light/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobile_light/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/mobilenetv1_light_from_buffer" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/mobile_light/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light_from_buffer/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/ssd_detection" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/ssd_detection/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/ssd_detection/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/yolov3_detection" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/yolov3_detection/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/yolov3_detection/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/mobile_classify" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/mobile_classify/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobile_classify/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/test_cv" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/test_cv/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/test_cv/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/mask_detection" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/mask_detection/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mask_detection/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/test_libs" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/test_libs/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/test_libs/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/quant_post_dynamic" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/quant_post_dynamic/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/quant_post_dynamic/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/lac_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/lac_demo/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/lac_demo/Makefile"
            )
            add_dependencies(publish_inference_android_cxx_demos gflags)
            add_dependencies(publish_inference_cxx_lib publish_inference_android_cxx_demos)
        else()
            # copy
            add_custom_target(publish_inference_android_cxx_demos ${TARGET}
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/Makefile.def" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/README.md" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/mobile_light" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/mobile_light/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobile_light/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/mobilenetv1_light_from_buffer" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/mobile_light/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light_from_buffer/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/ssd_detection" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/ssd_detection/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/ssd_detection/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/yolov3_detection" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/yolov3_detection/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/yolov3_detection/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/mobile_classify" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/mobile_classify/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobile_classify/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/test_cv" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/test_cv/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/test_cv/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/mask_detection" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/mask_detection/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mask_detection/Makefile"
                COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/lac_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/makefiles/lac_demo/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/lac_demo/Makefile"
            )
            add_dependencies(tiny_publish_cxx_lib publish_inference_android_cxx_demos)
        endif()

        if (LITE_WITH_JAVA)
            # copy java mobile_light demo/lib
            add_custom_target(publish_inference_android_java_demo ${TARGET}
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java"
                    COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/java/android" "${INFER_LITE_PUBLISH_ROOT}/demo/java"
                    COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/java/README.md" "${INFER_LITE_PUBLISH_ROOT}/demo/java"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/libs"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/src/main/jniLibs/arm7"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/src/main/jniLibs/arm8"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/src/main/jniLibs/arm64-v8a"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/src/main/jniLibs/armeabi-v7a"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/src/main/jniLibs/x86"
            )
            add_dependencies(publish_inference_java_lib publish_inference_android_java_demo)
        endif()
    endif()

    if (LITE_WITH_OPENCL)
        add_custom_target(publish_inference_opencl ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/opencl"
            COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/backends/opencl/cl_kernel" "${INFER_LITE_PUBLISH_ROOT}/opencl"
        )
       if (NOT LITE_ON_TINY_PUBLISH)
        add_dependencies(publish_inference_cxx_lib publish_inference_opencl)
       else()
        add_dependencies(tiny_publish_cxx_lib publish_inference_opencl)
       endif()
    endif()
endif()

if (LITE_WITH_METAL)
    add_custom_target(metal_lib_publish DEPENDS LiteMetalLIB
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/metal/"
            COMMAND cp -r "${PADDLE_BINARY_DIR}/lite.metallib" "${INFER_LITE_PUBLISH_ROOT}/metal/"
            COMMENT "COPY lite.metallib")

    if (NOT LITE_ON_TINY_PUBLISH)
        add_dependencies(publish_inference_cxx_lib metal_lib_publish)
    else ()
        add_dependencies(tiny_publish_cxx_lib metal_lib_publish)
    endif ()
endif ()

if(LITE_WITH_SW)
    add_custom_target(publish_inference_cxx_lib ${TARGET}
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/bin"
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
        COMMAND cp "${PADDLE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
        COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/paddle_use_kernels.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
        COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/paddle_use_ops.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
        COMMAND cp "${PADDLE_BINARY_DIR}/libpaddle_api_full_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
        COMMAND cp "${PADDLE_BINARY_DIR}/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
        COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/*.so" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
    )
    add_dependencies(publish_inference_cxx_lib bundle_full_api)
    add_dependencies(publish_inference_cxx_lib bundle_light_api)
    add_dependencies(publish_inference_cxx_lib paddle_full_api_shared)
    add_dependencies(publish_inference_cxx_lib paddle_light_api_shared)
    add_dependencies(publish_inference publish_inference_cxx_lib)
    add_dependencies(publish_inference test_model_bin)
endif()

if(LITE_WITH_XPU AND NOT XPU_SDK_ROOT)
    add_custom_command(TARGET publish_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${THIRD_PARTY_PATH}/install/xpu" "${INFER_LITE_PUBLISH_ROOT}/third_party/xpu")
endif()

if (ARM_TARGET_OS STREQUAL "armlinux")
    if (NOT LITE_ON_TINY_PUBLISH)
	add_custom_target(publish_inference_armlinux_cxx_demos ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
            COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/armlinux_mobilenetv1_full_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full"
            COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/armlinux_mobilenetv1_light_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light"
        )
    else()
	add_custom_target(publish_inference_armlinux_cxx_demos ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
            COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/armlinux_mobilenetv1_light_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light"
        )
    endif()
    add_dependencies(publish_inference publish_inference_armlinux_cxx_demos)
endif()

if (LITE_WITH_BM)
    add_custom_target(publish_inference_bm_cxx_demos ${TARGET}
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
        COMMAND cp "${PADDLE_SOURCE_DIR}/lite/demo/cxx/bm_demo/*" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/"
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx"
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib/third_party"
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
        COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/libpaddle_full_api_shared.so" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib/"
        COMMAND cp "${BM_SDK_ROOT}/lib/bmcompiler/*" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib/third_party/"
        COMMAND cp "${BM_SDK_ROOT}/lib/bmnn/pcie/*" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib/third_party/"
        COMMAND cp "${PADDLE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include/"
        COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/paddle_use_kernels.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
        COMMAND cp "${PADDLE_BINARY_DIR}/lite/api/paddle_use_ops.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
    )
    add_dependencies(publish_inference_bm_cxx_demos paddle_full_api_shared)
    add_dependencies(publish_inference publish_inference_bm_cxx_demos)
endif()

if (LITE_WITH_MLU)
    add_custom_target(publish_inference_mlu_cxx_demos ${TARGET}
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
        COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/demo/cxx/mlu_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mlu_demo/lib"
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mlu_demo/include"
        COMMAND cp -r "${PADDLE_BINARY_DIR}/lite/api/libpaddle_full_api_shared.so" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mlu_demo/lib"
        COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/api/paddle_place.h" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mlu_demo/include"
        COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/api/paddle_api.h" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mlu_demo/include"
    )
    add_dependencies(publish_inference_mlu_cxx_demos paddle_full_api_shared)
    add_dependencies(publish_inference publish_inference_mlu_cxx_demos)
endif()

if(LITE_WITH_NNADAPTER)
  # Build the NNAdapter runtime library, copy the header files and library file to the publish directory
  add_custom_target(publish_inference_nnadapter_runtime_header_files
      COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
      COMMAND cp -r "${PADDLE_SOURCE_DIR}/lite/backends/nnadapter/nnadapter/include/nnadapter" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
      DEPENDS nnadapter
      )
  add_dependencies(publish_inference publish_inference_nnadapter_runtime_header_files)
  add_custom_target(publish_inference_nnadapter_runtime_library_file
      COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
      COMMAND cp -r "${PADDLE_BINARY_DIR}/lite/backends/nnadapter/nnadapter/src/libnnadapter.so" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
      DEPENDS nnadapter
      )
  add_dependencies(publish_inference publish_inference_nnadapter_runtime_library_file)
  # Build the NNAdapter device HAL libraries for all of the specified devices and copy it to the publish directory
  foreach(device_name ${NNADAPTER_DEVICES})
    add_custom_target(publish_inference_nnadapter_${device_name}
      COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
      COMMAND cp -r "${PADDLE_BINARY_DIR}/lite/backends/nnadapter/nnadapter/src/driver/${device_name}/*.so" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
      DEPENDS ${device_name}
      )
    add_dependencies(publish_inference publish_inference_nnadapter_${device_name})
  endforeach()
endif()
