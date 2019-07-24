
lite_cc_library(npu_brigde_registry SRCS registry.cc DEPS ${npu_ddk_libs})

set(npu_brigde_deps ${npu_ddk_libs} npu_brigde_registry op)
lite_cc_library(npu_brigde_fc SRCS fc_op.cc DEPS ${npu_brigde_deps})
# lite_cc_library(npu_brigde_conv SRCS conv_op.cc DEPS ${npu_brigde_deps})

lite_cc_test(test_npu_bridge_fc SRCS fc_op_test.cc DEPS npu_brigde_fc)
# lite_cc_test(test_npu_bridge_conv SRCS conv_op_test.cc DEPS npu_brigde_fc)

set(npu_bridges
    npu_brigde_registry
    npu_brigde_fc)

set(npu_bridges ${npu_bridges} CACHE INTERNAL "npu_bridges")

message(STATUS "+++++ npu_bridges: ${npu_bridges}")
