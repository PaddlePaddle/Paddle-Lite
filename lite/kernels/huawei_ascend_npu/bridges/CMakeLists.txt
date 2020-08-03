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

set(huawei_ascend_npu_subgraph_bridges
        subgraph_bridge_registry
        subgraph_bridge_engine
        subgraph_bridge_graph_huawei_ascend_npu
        subgraph_bridge_act_op_huawei_ascend_npu
        subgraph_bridge_conv_op_huawei_ascend_npu
        subgraph_bridge_interpolate_op_huawei_ascend_npu
        subgraph_bridge_concat_op_huawei_ascend_npu
        CACHE INTERNAL "huawei_ascend_npu_subgraph_bridges")
