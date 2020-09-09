if(NOT LITE_WITH_IMAGINATION_NNA)
  return()
endif()

lite_cc_library(subgraph_bridge_utility_imagination_nna SRCS utility.cc DEPS ${imagination_nna_builder_libs} ${imagination_nna_runtime_libs} tensor)
lite_cc_library(subgraph_bridge_graph_imagination_nna SRCS graph.cc DEPS subgraph_bridge_utility_imagination_nna)

set(imagination_nna_subgraph_bridge_deps subgraph_bridge_registry subgraph_bridge_utility_imagination_nna subgraph_bridge_graph_imagination_nna)

lite_cc_library(subgraph_bridge_fc_op_imagination_nna SRCS fc_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_conv_op_imagination_nna SRCS conv_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_matmul_op_imagination_nna SRCS matmul_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_mul_op_imagination_nna SRCS mul_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_act_op_imagination_nna SRCS act_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_scale_op_imagination_nna SRCS scale_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_softmax_op_imagination_nna SRCS softmax_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_pool_op_imagination_nna SRCS pool_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_batch_norm_op_imagination_nna SRCS batch_norm_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_elementwise_ops_imagination_nna SRCS elementwise_ops.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_reshape_op_imagination_nna SRCS reshape_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_conv_transpose_op_imagination_nna SRCS conv_transpose_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_interpolate_op_imagination_nna SRCS interpolate_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_transpose_op_imagination_nna SRCS transpose_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_split_op_imagination_nna SRCS split_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_concat_op_imagination_nna SRCS concat_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_shuffle_channel_op_imagination_nna SRCS shuffle_channel_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_pad2d_op_imagination_nna SRCS pad2d_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_reduce_mean_op_imagination_nna SRCS reduce_mean_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_unsqueeze_op_imagination_nna SRCS unsqueeze_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_gather_op_imagination_nna SRCS gather_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_lookup_table_op_imagination_nna SRCS lookup_table_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_argmax_op_imagination_nna SRCS argmax_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_instance_norm_op_imagination_nna SRCS instance_norm_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_dropout_op_imagination_nna SRCS dropout_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_topk_op_imagination_nna SRCS topk_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_layer_norm_op_imagination_nna SRCS layer_norm_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_fill_constant_op_imagination_nna SRCS fill_constant_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_fill_constant_batch_size_like_op_imagination_nna SRCS fill_constant_batch_size_like_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_increment_op_imagination_nna SRCS increment_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_compare_op_imagination_nna SRCS compare_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})
#lite_cc_library(subgraph_bridge_shape_op_imagination_nna SRCS shape_op.cc DEPS ${imagination_nna_subgraph_bridge_deps})


set(imagination_nna_subgraph_bridges
        subgraph_bridge_registry
        subgraph_bridge_utility_imagination_nna
        subgraph_bridge_graph_imagination_nna
        subgraph_bridge_fc_op_imagination_nna
        subgraph_bridge_conv_op_imagination_nna
        #subgraph_bridge_matmul_op_imagination_nna
        #subgraph_bridge_mul_op_imagination_nna
        subgraph_bridge_act_op_imagination_nna
        #subgraph_bridge_scale_op_imagination_nna
        #subgraph_bridge_softmax_op_imagination_nna
        subgraph_bridge_pool_op_imagination_nna
        #subgraph_bridge_batch_norm_op_imagination_nna
        #subgraph_bridge_elementwise_ops_imagination_nna
        #subgraph_bridge_reshape_op_imagination_nna
        #subgraph_bridge_conv_transpose_op_imagination_nna
        #subgraph_bridge_interpolate_op_imagination_nna
        #subgraph_bridge_transpose_op_imagination_nna
        #subgraph_bridge_split_op_imagination_nna
        #subgraph_bridge_concat_op_imagination_nna
        #subgraph_bridge_shuffle_channel_op_imagination_nna
        #subgraph_bridge_pad2d_op_imagination_nna
        #subgraph_bridge_reduce_mean_op_imagination_nna
        #subgraph_bridge_unsqueeze_op_imagination_nna
        #subgraph_bridge_gather_op_imagination_nna
        #subgraph_bridge_lookup_table_op_imagination_nna
        #subgraph_bridge_argmax_op_imagination_nna
        #subgraph_bridge_instance_norm_op_imagination_nna
        #subgraph_bridge_dropout_op_imagination_nna
        #subgraph_bridge_topk_op_imagination_nna
        #subgraph_bridge_layer_norm_op_imagination_nna
        #subgraph_bridge_fill_constant_op_imagination_nna
        #subgraph_bridge_fill_constant_batch_size_like_op_imagination_nna
        #subgraph_bridge_increment_op_imagination_nna
        #subgraph_bridge_compare_op_imagination_nna
        CACHE INTERNAL "imagination_nna_subgraph_bridges")

message(STATUS "+++++ imagination_nna_subgraph_bridges: ${imagination_nna_subgraph_bridges}")
