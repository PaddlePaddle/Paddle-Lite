
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
      DEPS mir_pass types context ${mir_fusers} ${npu_bridges} npu_helper ${npu_ddk_libs} graph_op subgraph_pass)
  list(APPEND subgraph_passes npu_pass)
  lite_cc_test(test_npu_pass SRCS generate_npu_program_pass_test.cc
    DEPS npu_pass cxx_api mir_passes gflags
    ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v1
         --optimized_model=${LITE_MODEL_DIR}/lite_npu_model_opt SERIAL)
  if (WITH_TESTING)
    add_dependencies(test_npu_pass extern_lite_download_mobilenet_v1_tar_gz)
    add_dependencies(test_subgraph_pass extern_lite_download_mobilenet_v2_relu_tar_gz)
    set(LINK_FLAGS "-Wl,--version-script ${PADDLE_SOURCE_DIR}/lite/core/lite.map")
    set_target_properties(test_npu_pass PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
  endif()
endif()

set(subgraph_passes ${subgraph_passes} CACHE INTERNAL "subgraph_passes")
message(STATUS "----> subgraph_passes: ${subgraph_passes}")
