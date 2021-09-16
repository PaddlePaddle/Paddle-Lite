if (WITH_TESTING AND NOT LITE_WITH_CUDA AND NOT LITE_WITH_XPU)
    lite_cc_test(test_subgraph_detector
        SRCS subgraph_detector_test.cc
        ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v1 SERIAL)
    add_dependencies(test_subgraph_detector
        extern_lite_download_mobilenet_v1_tar_gz
        extern_lite_download_mobilenet_v2_relu_tar_gz)
    if(NOT WIN32)
      set(LINK_FLAGS "-Wl,--version-script ${PADDLE_SOURCE_DIR}/lite/core/lite.map")
      set_target_properties(test_subgraph_detector PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
    endif()
    lite_cc_test(test_subgraph_pass
        SRCS subgraph_pass_test.cc
        ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v1
             --optimized_model_dir=${LITE_MODEL_DIR}/lite_model_opt SERIAL)
    add_dependencies(test_subgraph_pass
        extern_lite_download_mobilenet_v1_tar_gz
        extern_lite_download_mobilenet_v2_relu_tar_gz)
    if(NOT WIN32)
        set(LINK_FLAGS "-Wl,--version-script ${PADDLE_SOURCE_DIR}/lite/core/lite.map")
        set_target_properties(test_subgraph_pass PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
    endif()
endif()
