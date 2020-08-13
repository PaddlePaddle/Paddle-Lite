lite_cc_library(fuse_fc
        SRCS fc_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_shuffle_channel
        SRCS shuffle_channel_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_conv_elementwise
        SRCS conv_elementwise_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_conv_activation
        SRCS conv_activation_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_var_conv_activation
        SRCS var_conv_2d_activation_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_conv_bn
        SRCS conv_bn_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_conv_conv
        SRCS conv_conv_fuser.cc
        DEPS pattern_matcher_high_api)     
lite_cc_library(fuse_elementwise_add_activation
        SRCS elementwise_add_activation_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_quant_dequant
        SRCS quant_dequant_op_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_transpose_softmax_transpose
        SRCS transpose_softmax_transpose_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_interpolate
        SRCS interpolate_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_sequence_pool_concat
        SRCS sequence_pool_concat_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_scale_activation
        SRCS scale_activation_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_reshape
        SRCS reshape_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_match_matrix_activation
        SRCS match_matrix_activation_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_scales
        SRCS scales_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_sequence_reverse_embedding
        SRCS sequence_reverse_embedding_fuser.cc
        DEPS pattern_matcher_high_api)

set(mir_fusers
    fuse_fc
    fuse_shuffle_channel
    fuse_conv_elementwise
    fuse_conv_activation
    fuse_var_conv_activation
    fuse_conv_bn
    fuse_conv_conv
    fuse_quant_dequant
    fuse_elementwise_add_activation
    fuse_transpose_softmax_transpose
    fuse_interpolate
    fuse_sequence_pool_concat
    fuse_scale_activation
    fuse_reshape
    fuse_match_matrix_activation
    fuse_scales
    fuse_sequence_reverse_embedding
    CACHE INTERNAL "fusers")

if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    return()
endif()

# TODO(Superjomn) Enable it latter
# NOTE disabled for the proto_desc is not valid yet.
# lite_cc_test(test_lite_conv_bn_fuse SRCS conv_bn_fuse_pass_test.cc
#    DEPS elementwise_ops batch_norm_op conv_op proto_desc compatible_pb program mir_pass mir_pass_manager pattern_matcher_high_api)
