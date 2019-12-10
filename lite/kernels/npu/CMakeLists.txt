
if(NOT LITE_WITH_NPU)
  return ()
endif()

message(STATUS "compile with lite NPU kernels")

add_subdirectory(bridges)

lite_cc_library(subgraph_engine_npu SRCS subgraph_engine.cc DEPS subgraph_engine_base device_npu ${npu_subgraph_bridges})

add_kernel(subgraph_compute_npu NPU basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} subgraph_engine_npu)
# lite_cc_test(test_subgraph_compute_npu SRCS subgraph_compute_test.cc DEPS subgraph_compute_npu)
