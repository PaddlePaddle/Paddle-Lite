add_subgraph_bridge(SRCS act_op.cc
                    argmax_op.cc
                    batch_norm_op.cc
                    cast_op.cc
                    concat_op.cc
                    conv_op.cc
                    dropout_op.cc
                    elementwise_ops.cc
                    fc_op.cc
                    flatten_op.cc
                    gather_op.cc
                    graph.cc
                    interpolate_op.cc
                    layer_norm_op.cc
                    matmul_op.cc
                    pool_op.cc
                    prior_box_op.cc
                    reduce_mean_op.cc
                    reshape_op.cc
                    scale_op.cc
                    shape_op.cc
                    slice_op.cc
                    softmax_op.cc
                    transpose_op.cc
                    unsqueeze_op.cc
                    utility.cc)

if(NOT LITE_WITH_HUAWEI_ASCEND_NPU)
  return()
endif()

lite_cc_library(subgraph_bridge_utility_huawei_ascend_npu SRCS utility.cc DEPS)
lite_cc_library(subgraph_bridge_graph_huawei_ascend_npu SRCS graph.cc DEPS subgraph_bridge_utility_huawei_ascend_npu)

set(huawei_ascend_npu_subgraph_bridge_deps core subgraph_bridge_utility_huawei_ascend_npu subgraph_bridge_graph_huawei_ascend_npu)

lite_cc_library(subgraph_bridge_act_op_huawei_ascend_npu SRCS act_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_conv_op_huawei_ascend_npu SRCS conv_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_interpolate_op_huawei_ascend_npu SRCS interpolate_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_concat_op_huawei_ascend_npu SRCS concat_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_pool_op_huawei_ascend_npu SRCS pool_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_elementwise_ops_huawei_ascend_npu SRCS elementwise_ops.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_batch_norm_op_huawei_ascend_npu SRCS batch_norm_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_softmax_op_huawei_ascend_npu SRCS softmax_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_dropout_op_huawei_ascend_npu SRCS dropout_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_fc_op_huawei_ascend_npu SRCS fc_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_reshape_op_huawei_ascend_npu SRCS reshape_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_transpose_op_huawei_ascend_npu SRCS transpose_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_flatten_op_huawei_ascend_npu SRCS flatten_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_layer_norm_op_huawei_ascend_npu SRCS layer_norm_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_matmul_op_huawei_ascend_npu SRCS matmul_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_cast_op_huawei_ascend_npu SRCS cast_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_scale_op_huawei_ascend_npu SRCS scale_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_slice_op_huawei_ascend_npu SRCS slice_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_gather_op_huawei_ascend_npu SRCS gather_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_prior_box_op_huawei_ascend_npu SRCS prior_box_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_argmax_op_huawei_ascend_npu SRCS argmax_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_shape_op_huawei_ascend_npu SRCS shape_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_unsqueeze_op_huawei_ascend_npu SRCS unsqueeze_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_reduce_mean_op_huawei_ascend_npu SRCS reduce_mean_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_clip_op_huawei_ascend_npu SRCS clip_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_fill_constant_op_huawei_ascend_npu SRCS fill_constant_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_split_op_huawei_ascend_npu SRCS split_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_assign_value_op_huawei_ascend_npu SRCS assign_value_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_norm_op_huawei_ascend_npu SRCS norm_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_pow_op_huawei_ascend_npu SRCS pow_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
lite_cc_library(subgraph_bridge_log_op_huawei_ascend_npu SRCS log_op.cc DEPS ${huawei_ascend_npu_subgraph_bridge_deps})
set(huawei_ascend_npu_subgraph_bridges
        core
        subgraph_bridge_graph_huawei_ascend_npu
        subgraph_bridge_act_op_huawei_ascend_npu
        subgraph_bridge_conv_op_huawei_ascend_npu
        subgraph_bridge_interpolate_op_huawei_ascend_npu
        subgraph_bridge_concat_op_huawei_ascend_npu
        subgraph_bridge_pool_op_huawei_ascend_npu
        subgraph_bridge_elementwise_ops_huawei_ascend_npu
        subgraph_bridge_batch_norm_op_huawei_ascend_npu
        subgraph_bridge_softmax_op_huawei_ascend_npu
        subgraph_bridge_dropout_op_huawei_ascend_npu
        subgraph_bridge_fc_op_huawei_ascend_npu
        subgraph_bridge_reshape_op_huawei_ascend_npu
        subgraph_bridge_transpose_op_huawei_ascend_npu
        subgraph_bridge_flatten_op_huawei_ascend_npu
        subgraph_bridge_layer_norm_op_huawei_ascend_npu
        subgraph_bridge_matmul_op_huawei_ascend_npu
        subgraph_bridge_cast_op_huawei_ascend_npu
        subgraph_bridge_scale_op_huawei_ascend_npu
        subgraph_bridge_slice_op_huawei_ascend_npu
        subgraph_bridge_gather_op_huawei_ascend_npu
        subgraph_bridge_prior_box_op_huawei_ascend_npu
        subgraph_bridge_argmax_op_huawei_ascend_npu
        subgraph_bridge_shape_op_huawei_ascend_npu
        subgraph_bridge_unsqueeze_op_huawei_ascend_npu
        subgraph_bridge_reduce_mean_op_huawei_ascend_npu
        subgraph_bridge_clip_op_huawei_ascend_npu
        subgraph_bridge_fill_constant_op_huawei_ascend_npu
        subgraph_bridge_split_op_huawei_ascend_npu
        subgraph_bridge_assign_value_op_huawei_ascend_npu
        subgraph_bridge_norm_op_huawei_ascend_npu
        subgraph_bridge_pow_op_huawei_ascend_npu
        subgraph_bridge_log_op_huawei_ascend_npu
        CACHE INTERNAL "huawei_ascend_npu_subgraph_bridges")
