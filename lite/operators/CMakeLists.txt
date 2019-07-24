set(op_DEPS tensor op op_params)

lite_cc_library(conv_op SRCS conv_op.cc DEPS ${op_DEPS})
lite_cc_library(pool_op SRCS pool_op.cc DEPS ${op_DEPS})
lite_cc_library(fc_op SRCS fc_op.cc DEPS ${op_DEPS})
lite_cc_library(relu_op SRCS relu_op.cc DEPS ${op_DEPS})
lite_cc_library(mul_op SRCS mul_op.cc DEPS ${op_DEPS})
lite_cc_library(scale_op SRCS scale_op.cc DEPS ${op_DEPS})
lite_cc_library(softmax_op SRCS softmax_op.cc DEPS ${op_DEPS})
lite_cc_library(reshape_op SRCS reshape_op.cc DEPS ${op_DEPS} )
lite_cc_library(batch_norm_op SRCS batch_norm_op.cc DEPS ${op_DEPS})
lite_cc_library(feed_op SRCS feed_op.cc DEPS ${op_DEPS})
lite_cc_library(fetch_op SRCS fetch_op.cc DEPS ${op_DEPS})
lite_cc_library(io_copy_op SRCS io_copy_op.cc DEPS ${op_DEPS})
lite_cc_library(activation_ops SRCS activation_ops.cc DEPS ${op_DEPS})
lite_cc_library(elementwise_ops SRCS elementwise_ops.cc DEPS ${op_DEPS})
lite_cc_library(fusion_elementwise_activation_ops SRCS fusion_elementwise_activation_ops.cc DEPS elementwise_ops ${op_DEPS})
lite_cc_library(mean_op SRCS mean_op.cc DEPS ${op_DEPS})
lite_cc_library(fill_constant_op SRCS fill_constant_op.cc DEPS ${op_DEPS})
lite_cc_library(sgd_op SRCS sgd_op.cc DEPS ${op_DEPS})
lite_cc_library(uniform_random_op SRCS uniform_random_op.cc DEPS ${op_DEPS})

lite_cc_library(op_params SRCS op_params.cc DEPS tensor any framework_proto)
lite_cc_library(dropout_op SRCS dropout_op.cc DEPS ${op_DEPS})
lite_cc_library(concat_op SRCS concat_op.cc DEPS ${op_DEPS})
lite_cc_library(normalize_op SRCS normalize_op.cc DEPS ${op_DEPS})
lite_cc_library(sroi_align_op SRCS sroi_align_op.cc DEPS ${op_DEPS})
lite_cc_library(pad2d_op SRCS pad2d_op.cc DEPS ${op_DEPS})
lite_cc_library(permute_op SRCS permute_op.cc DEPS ${op_DEPS})
lite_cc_library(negative_op SRCS negative_op.cc DEPS ${op_DEPS})
lite_cc_library(calib_op SRCS calib_op.cc DEPS ${op_DEPS})
lite_cc_library(split_op SRCS split_op.cc DEPS ${op_DEPS})
lite_cc_library(transpose_op SRCS transpose_op.cc DEPS ${op_DEPS})
lite_cc_library(fake_quant SRCS fake_quantize_moving_avg_max_abs.cc DEPS ${op_DEPS})
lite_cc_library(fake_dequant SRCS fake_dequantize_max_abs.cc DEPS ${op_DEPS})
lite_cc_library(im2sequence_op SRCS im2sequence_op.cc DEPS ${op_DEPS})
lite_cc_library(sequence_softmax_op SRCS sequence_softmax_op.cc DEPS ${op_DEPS})
lite_cc_library(norm_op SRCS norm_op.cc DEPS ${op_DEPS})
lite_cc_library(graph_op_lite SRCS graph_op.cc DEPS ${op_DEPS})

set(ops
        conv_op
        pool_op
        fc_op
        relu_op
        mul_op
        scale_op
        softmax_op
        reshape_op
        batch_norm_op
        feed_op
        fetch_op
        io_copy_op
        elementwise_ops
        fusion_elementwise_activation_ops
        mean_op
        fill_constant_op
        activation_ops
        dropout_op
        concat_op
        normalize_op
        sroi_align_op
        pad2d_op
        permute_op
        negative_op
        calib_op
        split_op
        transpose_op
        fake_quant
        fake_dequant
        sgd_op
        uniform_random_op
        im2sequence_op
        sequence_softmax_op
        norm_op
        CACHE INTERNAL "ops lite")

if (NOT LITE_WITH_X86)
    lite_cc_test(test_fc_op SRCS fc_op_test.cc
                DEPS fc_op memory
                X86_DEPS fc_compute_x86
                ARM_DEPS fc_compute_arm)
    lite_cc_test(test_pool_op SRCS pool_op_test.cc
                DEPS pool_op memory
                ARM_DEPS pool_compute_arm)
    lite_cc_test(test_scale_op SRCS scale_op_test.cc DEPS scale_op memory)
    lite_cc_test(test_softmax_op SRCS softmax_op_test.cc DEPS softmax_op memory)
    lite_cc_test(test_reshape_op SRCS reshape_op_test.cc DEPS reshape_op memory)
    lite_cc_test(test_batch_norm_op SRCS batch_norm_op_test.cc DEPS batch_norm_op memory)
    lite_cc_test(test_concat_op SRCS concat_op_test.cc DEPS concat_op memory)
    lite_cc_test(test_calib_op SRCS calib_op_test.cc DEPS calib_op memory ARM_DEPS calib_compute_arm)
    lite_cc_test(test_fusion_elementwise_activation_ops
                SRCS fusion_elementwise_activation_ops_test.cc
                DEPS fusion_elementwise_activation_ops memory)
    lite_cc_test(test_transpose_op SRCS transpose_op_test.cc DEPS transpose_op memory)
endif()
