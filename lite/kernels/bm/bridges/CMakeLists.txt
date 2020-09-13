if(NOT LITE_WITH_BM)
  return()
endif()

lite_cc_library(subgraph_bridge_utility_bm SRCS utility.cc DEPS)
lite_cc_library(subgraph_bridge_graph_bm SRCS graph.cc DEPS subgraph_bridge_utility_bm)

set(bm_subgraph_bridge_deps subgraph_bridge_registry subgraph_bridge_utility_bm subgraph_bridge_graph_bm)

lite_cc_library(subgraph_bridge_act_op_bm SRCS act_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_conv_op_bm SRCS conv_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_elementwise_ops_bm SRCS elementwise_ops.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_pool_op_bm SRCS pool_op.cc DEPS ${subgraph_bridge_deps_bm})
lite_cc_library(subgraph_bridge_softmax_op_bm SRCS softmax_op.cc DEPS ${subgraph_bridge_deps_bm})
lite_cc_library(subgraph_bridge_mul_op_bm SRCS mul_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_batch_norm_op_bm SRCS batch_norm_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_scale_op_bm SRCS scale_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_concat_op_bm SRCS concat_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_dropout_op_bm SRCS dropout_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_transpose_op_bm SRCS transpose_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_reshape_op_bm SRCS reshape_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_norm_op_bm SRCS norm_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_prior_box_op_bm SRCS prior_box_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_box_coder_op_bm SRCS box_coder_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_multiclass_nms_op_bm SRCS multiclass_nms_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_interpolate_op_bm SRCS interpolate_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_yolo_box_op_bm SRCS yolo_box_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_slice_op_bm SRCS slice_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_conv_transpose_op_bm SRCS conv_transpose_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_reduce_full_op_bm SRCS reduce_full_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_squeeze_op_bm SRCS squeeze_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_cast_op_bm SRCS cast_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_fill_constant_op_bm SRCS fill_constant_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_assign_value_op_bm SRCS assign_value_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_shape_op_bm SRCS shape_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_split_op_bm SRCS split_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_matmul_op_bm SRCS matmul_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_density_prior_box_op_bm SRCS density_prior_box_op.cc DEPS ${bm_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_swish_op_bm SRCS swish_op.cc DEPS ${bm_subgraph_bridge_deps})

set(bm_subgraph_bridges
        subgraph_bridge_registry
        subgraph_bridge_graph_bm
        subgraph_bridge_act_op_bm
        subgraph_bridge_conv_op_bm
        subgraph_bridge_elementwise_ops_bm
        subgraph_bridge_pool_op_bm
        subgraph_bridge_softmax_op_bm
        subgraph_bridge_mul_op_bm
        subgraph_bridge_batch_norm_op_bm
        subgraph_bridge_scale_op_bm
        subgraph_bridge_concat_op_bm
        subgraph_bridge_dropout_op_bm
        subgraph_bridge_transpose_op_bm
        subgraph_bridge_reshape_op_bm
        subgraph_bridge_norm_op_bm
        subgraph_bridge_prior_box_op_bm
        subgraph_bridge_box_coder_op_bm
        subgraph_bridge_multiclass_nms_op_bm
        subgraph_bridge_interpolate_op_bm
        subgraph_bridge_yolo_box_op_bm
        subgraph_bridge_slice_op_bm
        subgraph_bridge_conv_transpose_op_bm
        subgraph_bridge_reduce_full_op_bm
        subgraph_bridge_squeeze_op_bm
        subgraph_bridge_cast_op_bm
        subgraph_bridge_fill_constant_op_bm
        subgraph_bridge_assign_value_op_bm
        subgraph_bridge_shape_op_bm
        subgraph_bridge_split_op_bm
        subgraph_bridge_matmul_op_bm
        subgraph_bridge_density_prior_box_op_bm
        subgraph_bridge_swish_op_bm
        CACHE INTERNAL "bm_subgraph_bridges")
