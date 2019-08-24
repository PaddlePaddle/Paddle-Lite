
if(NOT LITE_WITH_NPU)
  return ()
endif()

message(STATUS "compile with lite NPU kernels")

add_kernel(graph_compute_npu NPU basic SRCS graph_compute.cc DEPS ${lite_kernel_deps} ${npu_ddk_libs})
# lite_cc_test(test_graph_compute_npu SRCS graph_compute_test.cc DEPS graph_compute_npu)
