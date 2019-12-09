lite_cc_library(npu_bridge_registry SRCS registry.cc)

set(npu_bridge_deps npu_bridge_registry npu_builder op)

lite_cc_library(npu_bridge_fc_op SRCS fc_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_conv_op SRCS conv_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_mul_op SRCS mul_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_act_op SRCS act_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_scale_op SRCS scale_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_softmax_op SRCS softmax_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_pool_op SRCS pool_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_batch_norm_op SRCS batch_norm_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_elementwise_ops SRCS elementwise_ops.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_reshape_op SRCS reshape_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_conv_transpose_op SRCS conv_transpose_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_interpolate_op SRCS interpolate_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_transpose_op SRCS transpose_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_split_op SRCS split_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_concat_op SRCS concat_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_shuffle_channel_op SRCS shuffle_channel_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_pad2d_op SRCS pad2d_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_square_op SRCS square_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_sqrt_op SRCS sqrt_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_reduce_mean_op SRCS reduce_mean_op.cc DEPS ${npu_bridge_deps})
lite_cc_library(npu_bridge_unsqueeze_op SRCS unsqueeze_op.cc DEPS ${npu_bridge_deps})

set(npu_bridges
        npu_bridge_registry
        npu_bridge_fc_op
        npu_bridge_conv_op
        npu_bridge_mul_op
        npu_bridge_act_op
        npu_bridge_scale_op
        npu_bridge_softmax_op
        npu_bridge_pool_op
        npu_bridge_batch_norm_op
        npu_bridge_elementwise_ops
        npu_bridge_reshape_op
        npu_bridge_conv_transpose_op
        npu_bridge_interpolate_op
        npu_bridge_transpose_op
        npu_bridge_split_op
        npu_bridge_concat_op
        npu_bridge_shuffle_channel_op
        npu_bridge_pad2d_op
        npu_bridge_square_op
        npu_bridge_sqrt_op
        npu_bridge_reduce_mean_op
        npu_bridge_unsqueeze_op
        CACHE INTERNAL "npu_bridges")

set(npu_bridge_test_deps ${npu_bridges} ${npu_kernels} ${ops})

lite_cc_test(test_npu_bridge_fc_op SRCS fc_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_conv_op SRCS conv_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_mul_op SRCS mul_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_act_op SRCS act_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_scale_op SRCS scale_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_softmax_op SRCS softmax_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_pool_op SRCS pool_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_batch_norm_op SRCS batch_norm_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_elementwise_ops SRCS elementwise_ops_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_reshape_op SRCS reshape_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_conv_transpose_op SRCS conv_transpose_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_interpolate_op SRCS interpolate_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_transpose_op SRCS transpose_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_split_op SRCS split_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_concat_op SRCS concat_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_shuffle_channel_op SRCS shuffle_channel_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_pad2d_op SRCS pad2d_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_square_op SRCS square_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_sqrt_op SRCS sqrt_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_reduce_mean_op SRCS reduce_mean_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})
lite_cc_test(test_npu_bridge_unsqueeze_op SRCS unsqueeze_op_test.cc test_helper.cc DEPS ${npu_bridge_test_deps})

message(STATUS "+++++ npu_bridges: ${npu_bridges}")
