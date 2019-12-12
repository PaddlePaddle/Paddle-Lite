
lite_cc_library(subgraph_pass
    SRCS subgraph_program_pass.cc
    DEPS mir_pass types ${mir_fusers})
lite_cc_test(test_subgraph_pass SRCS subgraph_program_pass_test.cc
  DEPS subgraph_pass mir_passes gflags model_parser cxx_api
  ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v1 SERIAL)
if (WITH_TESTING)
  add_dependencies(test_subgraph_pass extern_lite_download_mobilenet_v1_tar_gz)
  add_dependencies(test_subgraph_pass extern_lite_download_mobilenet_v2_relu_tar_gz)
  set(LINK_FLAGS "-Wl,--version-script ${PADDLE_SOURCE_DIR}/lite/core/lite.map")
  set_target_properties(test_subgraph_pass PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
endif()

set(subgraph_passes subgraph_pass)

if(LITE_WITH_NPU)
  lite_cc_library(npu_pass SRCS generate_npu_program_pass.cc
      DEPS mir_pass types context ${mir_fusers} ${npu_bridges} graph_op subgraph_pass)
  list(APPEND subgraph_passes npu_pass)
  lite_cc_test(test_npu_pass SRCS generate_npu_program_pass_test.cc
    DEPS npu_pass mir_passes paddle_api_full paddle_api_light gflags
    ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v1
         --optimized_model=${LITE_MODEL_DIR}/lite_npu_model_opt SERIAL)
  if (WITH_TESTING)
    add_dependencies(test_npu_pass extern_lite_download_mobilenet_v1_tar_gz)
    add_dependencies(test_subgraph_pass extern_lite_download_mobilenet_v2_relu_tar_gz)
    set(LINK_FLAGS "-Wl,--version-script ${PADDLE_SOURCE_DIR}/lite/core/lite.map")
    set_target_properties(test_npu_pass PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
  endif()
endif()

if(LITE_WITH_XPU)
  lite_cc_library(xpu_pass SRCS generate_xpu_program_pass.cc
      DEPS mir_pass types context ${mir_fusers} ${xpu_bridges} ${xpu_builder_libs} graph_op subgraph_pass)
  list(APPEND subgraph_passes xpu_pass)
  lite_cc_test(test_xpu_pass SRCS generate_xpu_program_pass_test.cc
    DEPS xpu_pass mir_passes paddle_api_full gflags
    ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v1
         --optimized_model=${LITE_MODEL_DIR}/lite_npu_model_opt SERIAL)
  if (WITH_TESTING)
    add_dependencies(test_xpu_pass extern_lite_download_mobilenet_v1_tar_gz)
    add_dependencies(test_subgraph_pass extern_lite_download_mobilenet_v2_relu_tar_gz)
    set(LINK_FLAGS "-Wl,--version-script ${PADDLE_SOURCE_DIR}/lite/core/lite.map")
    set_target_properties(test_xpu_pass PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
  endif()
endif()

if(LITE_WITH_BM)
  lite_cc_library(bm_pass SRCS generate_bm_program_pass.cc
      DEPS mir_pass types context ${mir_fusers} ${bm_bridges} ${bm_builder_libs} graph_op subgraph_pass)
  list(APPEND subgraph_passes bm_pass)
endif()

set(subgraph_passes ${subgraph_passes} CACHE INTERNAL "subgraph_passes")
message(STATUS "----> subgraph_passes: ${subgraph_passes}")
