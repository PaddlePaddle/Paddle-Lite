
if(NOT LITE_WITH_NPU)
  return ()
endif()

message(STATUS "compile with lite NPU kernels")

add_kernel(subgraph_compute_npu NPU basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} npu_runtime)
# lite_cc_test(test_subgraph_compute_npu SRCS subgraph_compute_test.cc DEPS subgraph_compute_npu)

if(NOT LITE_ON_TINY_PUBLISH)
    add_subdirectory(bridges)
endif()
