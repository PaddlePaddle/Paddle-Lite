lite_cc_library(bm_bridge_registry SRCS registry.cc)

set(bm_bridge_deps bm_bridge_registry op)

lite_cc_library(bm_bridge_act_op SRCS act_op.cc DEPS ${bm_bridge_deps})
lite_cc_library(bm_bridge_conv_op SRCS conv_op.cc DEPS ${bm_bridge_deps})
lite_cc_library(bm_bridge_elementwise_ops SRCS elementwise_ops.cc DEPS ${bm_bridge_deps})
lite_cc_library(bm_bridge_pool_op SRCS pool_op.cc DEPS ${bm_bridge_deps})
lite_cc_library(bm_bridge_softmax_op SRCS softmax_op.cc DEPS ${bm_bridge_deps})
lite_cc_library(bm_bridge_mul_op SRCS mul_op.cc DEPS ${bm_bridge_deps})
lite_cc_library(bm_bridge_batch_norm_op SRCS batch_norm_op.cc DEPS ${bm_bridge_deps})

set(bm_bridges
        bm_bridge_registry
        bm_bridge_act_op
        bm_bridge_conv_op
        bm_bridge_elementwise_ops
        bm_bridge_pool_op
        bm_bridge_softmax_op
        bm_bridge_mul_op
        bm_bridge_batch_norm_op
        CACHE INTERNAL "bm_bridges")

