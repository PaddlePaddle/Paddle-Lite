
if(NOT LITE_WITH_NPU)
  return ()
endif()

message(STATUS "compile with lite NPU kernels")

lite_cc_library(npu_runtime SRCS runtime.cc DEPS npu_ddk_hiai)

add_kernel(graph_compute_npu NPU basic SRCS graph_compute.cc DEPS ${lite_kernel_deps} npu_runtime)
# lite_cc_test(test_graph_compute_npu SRCS graph_compute_test.cc DEPS graph_compute_npu)

if(NOT LITE_ON_TINY_PUBLISH)
    add_subdirectory(bridges)
endif()
