
if(NOT LITE_WITH_XPU)
  return ()
endif()

add_kernel(subgraph_compute_xpu XPU basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} xpu_runtime)
# lite_cc_test(test_subgraph_compute_xpu SRCS subgraph_compute_test.cc DEPS subgraph_compute_xpu)

add_subdirectory(bridges)
