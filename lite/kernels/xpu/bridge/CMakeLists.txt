lite_cc_library(xpu_bridge_registry SRCS registry.cc)

set(xpu_bridge_deps xpu_bridge_registry xpu_builder op)

lite_cc_library(xpu_bridge_act_op SRCS act_op.cc DEPS ${xpu_bridge_deps})

set(xpu_bridges
        xpu_bridge_registry
        xpu_bridge_act_op
        CACHE INTERNAL "xpu_bridges")

set(xpu_bridge_test_deps ${xpu_bridges} ${xpu_kernels} ${ops})

lite_cc_test(test_xpu_bridge_act_op SRCS act_op_test.cc test_helper.cc DEPS ${xpu_bridge_test_deps})
