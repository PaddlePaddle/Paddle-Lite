
set(subgraph_passes)

if(LITE_WITH_NPU)
  lite_cc_library(npu_pass SRCS generate_npu_program_pass.cc
      DEPS mir_pass types context ${mir_fusers} ${npu_bridges} npu_helper ${npu_ddk_libs})
  list(APPEND subgraph_passes npu_pass)
  # lite_cc_test(test_npu_pass SRCS DEPS)
endif()

set(subgraph_passes ${subgraph_passes} CACHE INTERNAL "subgraph_passes")

message(STATUS "+++++ subgraph_passes: ${subgraph_passes}")
