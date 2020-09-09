if(NOT LITE_WITH_NPU)
  return()
endif()

lite_cc_library(subgraph_bridge_utility_npu SRCS utility.cc DEPS ${npu_builder_libs} tensor)
lite_cc_library(subgraph_bridge_graph_npu SRCS graph.cc DEPS subgraph_bridge_utility_npu)

set(npu_subgraph_bridge_deps subgraph_bridge_registry subgraph_bridge_utility_npu subgraph_bridge_graph_npu)

lite_cc_library(subgraph_bridge_fc_op_npu SRCS fc_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_conv_op_npu SRCS conv_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_matmul_op_npu SRCS matmul_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_mul_op_npu SRCS mul_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_act_op_npu SRCS act_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_scale_op_npu SRCS scale_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_softmax_op_npu SRCS softmax_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_pool_op_npu SRCS pool_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_batch_norm_op_npu SRCS batch_norm_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_elementwise_ops_npu SRCS elementwise_ops.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_reshape_op_npu SRCS reshape_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_conv_transpose_op_npu SRCS conv_transpose_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_interpolate_op_npu SRCS interpolate_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_transpose_op_npu SRCS transpose_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_split_op_npu SRCS split_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_concat_op_npu SRCS concat_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_shuffle_channel_op_npu SRCS shuffle_channel_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_pad2d_op_npu SRCS pad2d_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_reduce_mean_op_npu SRCS reduce_mean_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_unsqueeze_op_npu SRCS unsqueeze_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_gather_op_npu SRCS gather_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_lookup_table_op_npu SRCS lookup_table_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_argmax_op_npu SRCS argmax_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_instance_norm_op_npu SRCS instance_norm_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_dropout_op_npu SRCS dropout_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_topk_op_npu SRCS topk_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_layer_norm_op_npu SRCS layer_norm_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_fill_constant_op_npu SRCS fill_constant_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_fill_constant_batch_size_like_op_npu SRCS fill_constant_batch_size_like_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_increment_op_npu SRCS increment_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_compare_op_npu SRCS compare_op.cc DEPS ${npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_expand_op_npu SRCS expand_op.cc DEPS ${npu_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_shape_op_npu SRCS shape_op.cc DEPS ${npu_subgraph_bridge_deps})


set(npu_subgraph_bridges
        subgraph_bridge_registry
        subgraph_bridge_utility_npu
        subgraph_bridge_graph_npu
        subgraph_bridge_fc_op_npu
        subgraph_bridge_conv_op_npu
        subgraph_bridge_matmul_op_npu
        subgraph_bridge_mul_op_npu
        subgraph_bridge_act_op_npu
        subgraph_bridge_scale_op_npu
        subgraph_bridge_softmax_op_npu
        subgraph_bridge_pool_op_npu
        subgraph_bridge_batch_norm_op_npu
        subgraph_bridge_elementwise_ops_npu
        subgraph_bridge_reshape_op_npu
        subgraph_bridge_conv_transpose_op_npu
        subgraph_bridge_interpolate_op_npu
        subgraph_bridge_transpose_op_npu
        subgraph_bridge_split_op_npu
        subgraph_bridge_concat_op_npu
        subgraph_bridge_shuffle_channel_op_npu
        subgraph_bridge_pad2d_op_npu
        subgraph_bridge_reduce_mean_op_npu
        subgraph_bridge_unsqueeze_op_npu
        subgraph_bridge_gather_op_npu
        subgraph_bridge_lookup_table_op_npu
        subgraph_bridge_argmax_op_npu
        subgraph_bridge_instance_norm_op_npu
        subgraph_bridge_dropout_op_npu
        subgraph_bridge_topk_op_npu
        subgraph_bridge_layer_norm_op_npu
        subgraph_bridge_fill_constant_op_npu
        subgraph_bridge_fill_constant_batch_size_like_op_npu
        subgraph_bridge_increment_op_npu
        subgraph_bridge_compare_op_npu
        subgraph_bridge_expand_op_npu
        CACHE INTERNAL "npu_subgraph_bridges")

message(STATUS "+++++ npu_subgraph_bridges: ${npu_subgraph_bridges}")
