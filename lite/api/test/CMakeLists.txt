include(ExternalProject)
set(LITE_DEMO_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo" CACHE STRING
        "A path setting inference demo download directories.")

if(WITH_TESTING)
    if(NOT WITH_COVERAGE)
        lite_cc_test(test_cxx_api SRCS cxx_api_test.cc
           EXCLUDE_COMPILE_DEPS "ON"
           ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model
                --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)
        add_dependencies(test_cxx_api extern_lite_download_lite_naive_model_tar_gz)
    endif()
    if(NOT LITE_WITH_ARM)
        if(LITE_WITH_X86)
            lite_cc_test(test_googlenet SRCS test_googlenet_lite.cc
               ARGS --model_dir=${LITE_MODEL_DIR}/googlenet)
            add_dependencies(test_googlenet extern_lite_download_GoogleNet_inference_tar_gz)
            lite_cc_test(test_mobilenetv1_lite_x86 SRCS test_mobilenetv1_lite_x86.cc
               ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v1)
            if(LITE_WITH_METAL)
                lite_cc_test(test_mobilenetv1_lite_metal SRCS test_mobilenetv1_lite_metal.cc
                        ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v1 --metal_dir=${PADDLE_BINARY_DIR}/lite/backends/metal/lite.metallib)
            endif()
            add_dependencies(test_mobilenetv1_lite_x86 extern_lite_download_mobilenet_v1_tar_gz)
            lite_cc_test(test_mobilenetv2_lite_x86 SRCS test_mobilenetv2_lite_x86.cc
               ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v2_relu)
            add_dependencies(test_mobilenetv2_lite_x86 extern_lite_download_mobilenet_v2_relu_tar_gz)
            lite_cc_test(test_inceptionv4_lite_x86 SRCS test_inceptionv4_lite_x86.cc
               ARGS --model_dir=${LITE_MODEL_DIR}/inception_v4_simple)
            add_dependencies(test_inceptionv4_lite_x86 extern_lite_download_inception_v4_simple_tar_gz)
            lite_cc_test(test_resnet50_lite_x86 SRCS test_resnet50_lite_x86.cc
               ARGS --model_dir=${LITE_MODEL_DIR}/resnet50)
            add_dependencies(test_resnet50_lite_x86 extern_lite_download_resnet50_tar_gz)
            lite_cc_test(test_step_rnn_lite_x86 SRCS test_step_rnn_lite_x86.cc
               ARGS --model_dir=${LITE_MODEL_DIR}/step_rnn)
            add_dependencies(test_step_rnn_lite_x86 extern_lite_download_step_rnn_tar_gz)
        endif()
        if(LITE_WITH_BM)
           lite_cc_test(test_classify_lite_bm SRCS test_classify_lite_bm.cc
              ARGS --model_dir=${LITE_MODEL_DIR}/classify)
           lite_cc_test(test_yolov3_lite_bm SRCS test_yolov3_lite_bm.cc
              ARGS --model_dir=${LITE_MODEL_DIR}/yolov3)
        endif()
    endif()
endif()

if(LITE_WITH_ARM AND WITH_TESTING)
    set(lite_model_test_DEPS cxx_api ops kernels)

    lite_cc_test(test_mobilenetv1_int8 SRCS mobilenetv1_int8_test.cc
       ARGS --model_dir=${LITE_MODEL_DIR}/MobilenetV1_quant SERIAL)
    add_dependencies(test_mobilenetv1_int8 extern_lite_download_MobileNetV1_quant_tar_gz)

    lite_cc_test(test_mobilenetv1 SRCS mobilenetv1_test.cc
       ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v1 SERIAL)
    add_dependencies(test_mobilenetv1 extern_lite_download_mobilenet_v1_tar_gz)
    if(NOT WIN32)
       set(LINK_FLAGS "-Wl,--version-script ${PADDLE_SOURCE_DIR}/lite/core/lite.map")
       set_target_properties(test_mobilenetv1 PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
    endif()
    
    lite_cc_test(test_mobilenetv1_int16 SRCS mobilenetv1_int16_test.cc
       ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v1_int16 SERIAL)
    add_dependencies(test_mobilenetv1 extern_lite_download_mobilenet_v1_int16_tar_gz)
    
    lite_cc_test(test_mobilenetv1_opt_quant SRCS mobilenetv1_opt_quant_test.cc
       ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v1 SERIAL)
    add_dependencies(test_mobilenetv1 extern_lite_download_mobilenet_v1_tar_gz)

    lite_cc_test(test_mobilenetv2 SRCS mobilenetv2_test.cc
       ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v2_relu SERIAL)
    add_dependencies(test_mobilenetv2 extern_lite_download_mobilenet_v2_relu_tar_gz)
    if(NOT WIN32)
        set_target_properties(test_mobilenetv2 PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
    endif()

    lite_cc_test(test_resnet50 SRCS resnet50_test.cc
       ARGS --model_dir=${LITE_MODEL_DIR}/resnet50 SERIAL)
    add_dependencies(test_resnet50 extern_lite_download_resnet50_tar_gz)

    lite_cc_test(test_resnet50_fpga SRCS resnet50_test_fpga.cc)

    lite_cc_test(test_inceptionv4 SRCS inceptionv4_test.cc
       ARGS --model_dir=${LITE_MODEL_DIR}/inception_v4 SERIAL)
    add_dependencies(test_inceptionv4 extern_lite_download_inception_v4_simple_tar_gz)

    lite_cc_test(test_fast_rcnn SRCS fast_rcnn_test.cc)
    add_dependencies(test_fast_rcnn extern_lite_download_fast_rcnn_fluid184_tar_gz)

   # brief: we comment ocr_test_ut because we do not supply ocr model to test, it is the reference to infer nlp model
   # lite_cc_test(test_ocr_attention SRCS ocr_attention_test.cc)

   # lite_cc_test(model_run_test_image SRCS model_run_test_image.cc)

endif()

#-----------------------------------------------------------------------------------------------------
# These tests needs CLI arguments, and is not supported in ARM CI.
# TODO(Superjomn) support latter.
if(NOT WITH_COVERAGE)
    lite_cc_test(test_light_api SRCS light_api_test.cc
        ARGS --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)

    lite_cc_test(test_apis SRCS apis_test.cc
        ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model
        --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)
endif()

if (LITE_ON_TINY_PUBLISH)
    return()
endif()



if(NOT WITH_COVERAGE)
    lite_cc_test(test_paddle_api SRCS paddle_api_test.cc
      ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model SERIAL)
    if (WITH_TESTING)
        add_dependencies(test_paddle_api extern_lite_download_lite_naive_model_tar_gz)
    endif()
endif()

# Some bins
if(NOT IOS)
    lite_cc_binary(test_model_detection_bin SRCS model_test_detection.cc
        CV_DEPS paddle_cv_arm)
    
    lite_cc_binary(test_model_classify_bin SRCS model_test_classify.cc
        CV_DEPS paddle_cv_arm)

    lite_cc_binary(multithread_test SRCS lite_multithread_test.cc)

    lite_cc_binary(test_transformer SRCS transform_test.cc)
endif()
