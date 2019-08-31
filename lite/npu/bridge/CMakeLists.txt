
lite_cc_library(npu_bridge_registry SRCS registry.cc DEPS ${npu_ddk_libs})
lite_cc_library(npu_bridge_utils SRCS utils.cc DEPS ${npu_ddk_libs} tensor op mir_node scope)

set(npu_bridge_deps npu_bridge_registry npu_bridge_utils op)

lite_cc_library(npu_bridge_fc_op SRCS fc_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_conv_op SRCS conv_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_mul_op SRCS mul_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_act_op SRCS act_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_scale_op SRCS scale_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_softmax_op SRCS softmax_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_pool_op SRCS pool_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_batch_norm_op SRCS batch_norm_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_elementwise_op SRCS elementwise_ops.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_reshape_op SRCS reshape_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_conv_transpose_op SRCS conv_transpose_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_interpolate_op SRCS interpolate_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_transpose_op SRCS transpose_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_split_op SRCS split_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_concat_op SRCS concat_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_shuffle_channel_op SRCS shuffle_channel_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_pad2d_op SRCS pad2d_op.cc DEPS ${npu_bridge_deps})

set(npu_bridges
        npu_bridge_registry
        npu_bridge_utils
        npu_bridge_fc_op
        npu_bridge_conv_op
        npu_bridge_mul_op
        npu_bridge_act_op
        npu_bridge_scale_op
        npu_bridge_softmax_op
        npu_bridge_pool_op
        npu_bridge_batch_norm_op
        npu_bridge_elementwise_op
        npu_bridge_reshape_op
        npu_bridge_conv_transpose_op
        npu_bridge_interpolate_op
        npu_bridge_transpose_op
        npu_bridge_split_op
        npu_bridge_concat_op
        npu_bridge_shuffle_channel_op
        npu_bridge_pad2d_op
        CACHE INTERNAL "npu_bridges")

lite_cc_library(npu_test_helper SRCS test_helper.cc DEPS npu_helper ${npu_ddk_libs} ${npu_bridges} ${npu_kernels} ${ops})

lite_cc_test(test_npu_bridge_fc_op SRCS fc_op_test.cc DEPS npu_test_helper)
lite_cc_test(test_npu_bridge_conv_op SRCS conv_op_test.cc DEPS npu_test_helper)
lite_cc_test(test_npu_bridge_mul_op SRCS mul_op_test.cc DEPS npu_test_helper)
lite_cc_test(test_npu_bridge_act_op SRCS act_op_test.cc DEPS npu_test_helper)
lite_cc_test(test_npu_bridge_scale_op SRCS scale_op_test.cc DEPS npu_test_helper)
lite_cc_test(test_npu_bridge_softmax_op SRCS softmax_op_test.cc DEPS npu_test_helper)
lite_cc_test(test_npu_bridge_pool_op SRCS pool_op_test.cc DEPS npu_test_helper)
lite_cc_test(test_npu_bridge_batch_norm_op SRCS batch_norm_op_test.cc DEPS npu_test_helper)
lite_cc_test(test_npu_bridge_elementwise_op SRCS elementwise_ops_test.cc DEPS npu_test_helper)
lite_cc_test(test_npu_bridge_reshape_op SRCS reshape_op_test.cc DEPS npu_test_helper)
lite_cc_test(test_npu_bridge_conv_transpose_op SRCS conv_transpose_op_test.cc DEPS npu_test_helper)
lite_cc_test(test_npu_bridge_interpolate_op SRCS interpolate_op_test.cc DEPS npu_test_helper)
lite_cc_test(test_npu_bridge_transpose_op SRCS transpose_op_test.cc DEPS npu_test_helper)
lite_cc_test(test_npu_bridge_split_op SRCS split_op_test.cc DEPS npu_test_helper)
lite_cc_test(test_npu_bridge_concat_op SRCS concat_op_test.cc DEPS npu_test_helper)
lite_cc_test(test_npu_bridge_shuffle_channel_op SRCS shuffle_channel_op_test.cc DEPS npu_test_helper)
lite_cc_test(test_npu_bridge_pad2d_op SRCS pad2d_op_test.cc DEPS npu_test_helper)

message(STATUS "+++++ npu_bridges: ${npu_bridges}")
