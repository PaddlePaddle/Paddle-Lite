
if(NOT LITE_WITH_XPU)
  return ()
endif()

add_kernel(graph_compute_xpu XPU basic SRCS graph_compute.cc DEPS ${lite_kernel_deps} xpu_runtime)
# lite_cc_test(test_graph_compute_xpu SRCS graph_compute_test.cc DEPS graph_compute_xpu)

add_subdirectory(bridges)
