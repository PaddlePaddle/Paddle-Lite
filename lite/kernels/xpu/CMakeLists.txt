
if(NOT LITE_WITH_XPU)
  return ()
endif()

message(STATUS "compile with lite XPU kernels")

add_subdirectory(bridges)

lite_cc_library(subgraph_engine_xpu SRCS subgraph_engine.cc DEPS subgraph_engine_base device_xpu ${xpu_subgraph_bridges})

add_kernel(subgraph_compute_xpu XPU basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} subgraph_engine_xpu)
# lite_cc_test(test_subgraph_compute_xpu SRCS subgraph_compute_test.cc DEPS subgraph_compute_xpu)
