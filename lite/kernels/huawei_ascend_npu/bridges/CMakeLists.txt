if(NOT LITE_WITH_HUAWEI_ASCEND_NPU)
  return()
endif()

lite_cc_library(subgraph_bridge_utility_huawei_ascend_npu SRCS utility.cc DEPS)
lite_cc_library(subgraph_bridge_graph_huawei_ascend_npu SRCS graph.cc DEPS subgraph_bridge_utility_huawei_ascend_npu)

set(huawei_ascend_npu_subgraph_bridge_deps subgraph_bridge_registry subgraph_bridge_engine subgraph_bridge_utility_huawei_ascend_npu subgraph_bridge_graph_huawei_ascend_npu)

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

set(huawei_ascend_npu_subgraph_bridges
        subgraph_bridge_registry
        subgraph_bridge_engine
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
        CACHE INTERNAL "huawei_ascend_npu_subgraph_bridges")
