
lite_cc_library(npu_brigde_registry SRCS registry.cc DEPS ${npu_ddk_libs})
lite_cc_library(npu_brigde_utils SRCS utils.cc DEPS ${npu_ddk_libs})

set(npu_brigde_deps npu_brigde_registry npu_brigde_utils op)
lite_cc_library(npu_brigde_fc_op SRCS fc_op.cc DEPS ${npu_brigde_deps})
lite_cc_library(npu_brigde_conv_op SRCS conv_op.cc DEPS ${npu_brigde_deps})
lite_cc_library(npu_brigde_mul_op SRCS mul_op.cc DEPS ${npu_brigde_deps})
lite_cc_library(npu_brigde_act_op SRCS act_op.cc DEPS ${npu_brigde_deps})
lite_cc_library(npu_brigde_scale_op SRCS scale_op.cc DEPS ${npu_brigde_deps})

set(npu_brigdes
        npu_brigde_registry
        npu_brigde_utils
        npu_brigde_fc_op
        npu_brigde_conv_op
        npu_brigde_mul_op
        npu_brigde_act_op
        npu_brigde_scale_op
        CACHE INTERNAL "npu_brigdes")

lite_cc_test(test_npu_bridge_fc_op SRCS fc_op_test.cc DEPS npu_brigde_fc_op ${ops})
lite_cc_test(test_npu_bridge_conv_op SRCS conv_op_test.cc DEPS npu_brigde_conv_op ${ops})
lite_cc_test(test_npu_bridge_mul_op SRCS mul_op_test.cc DEPS npu_brigde_mul_op ${ops})
lite_cc_test(test_npu_bridge_act_op SRCS act_op_test.cc DEPS npu_brigde_act_op ${ops})
lite_cc_test(test_npu_bridge_scale_op SRCS scale_op_test.cc DEPS npu_brigde_scale_op ${ops})

message(STATUS "+++++ npu_brigdes: ${npu_brigdes}")
