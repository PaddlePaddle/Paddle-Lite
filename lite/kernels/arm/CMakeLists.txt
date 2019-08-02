if(NOT (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM))
    return()
endif()

message(STATUS "compile with lite ARM kernels")

lite_cc_library(fc_compute_arm SRCS fc_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(activation_compute_arm SRCS activation_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(mul_compute_arm SRCS mul_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(scale_compute_arm SRCS scale_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(softmax_compute_arm SRCS softmax_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(conv_compute_arm SRCS conv_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(batch_norm_compute_arm SRCS batch_norm_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(elementwise_compute_arm SRCS elementwise_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(lrn_compute_arm SRCS lrn_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(decode_bboxes_compute_arm SRCS decode_bboxes_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(multiclass_nms_compute_arm SRCS multiclass_nms_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(pool_compute_arm SRCS pool_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(split_compute_arm SRCS split_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(concat_compute_arm SRCS concat_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(pad2d_compute_arm SRCS pad2d_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(prior_box_compute_arm SRCS prior_box_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(density_prior_box_compute_arm SRCS density_prior_box_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(negative_compute_arm SRCS negative_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(crop_compute_arm SRCS crop_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(dropout_compute_arm SRCS dropout_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(calib_compute_arm SRCS calib_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(transpose_compute_arm SRCS transpose_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(argmax_compute_arm SRCS argmax_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(axpy_compute_arm SRCS axpy_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(conv_transpose_compute_arm SRCS conv_transpose_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(gru_unit_compute_arm SRCS gru_unit_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(gru_compute_arm SRCS gru_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(beam_search_decode_compute_arm SRCS beam_search_decode_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(im2sequence_compute_arm SRCS im2sequence_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(sequence_softmax_compute_arm SRCS sequence_softmax_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(norm_compute_arm SRCS norm_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(logical_compute_arm SRCS logical_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(compare_compute_arm SRCS compare_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(topk_compute_arm SRCS topk_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(increment_compute_arm SRCS increment_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(write_to_array_compute_arm SRCS write_to_array_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(read_from_array_compute_arm SRCS read_from_array_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(beam_search_compute_arm SRCS beam_search_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(box_coder_compute_arm SRCS box_coder_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(sequence_pool_compute_arm SRCS sequence_pool_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(sequence_expand_compute_arm SRCS sequence_expand_compute.cc DEPS ${lite_kernel_deps} math_arm)
lite_cc_library(reduce_max_compute_arm SRCS reduce_max_compute.cc DEPS ${lite_kernel_deps} math_arm)

lite_cc_test(test_fc_compute_arm SRCS fc_compute_test.cc DEPS fc_compute_arm math_arm)
lite_cc_test(test_activation_compute_arm SRCS activation_compute_test.cc DEPS activation_compute_arm)
lite_cc_test(test_scale_compute_arm SRCS scale_compute_test.cc DEPS scale_compute_arm)
lite_cc_test(test_softmax_compute_arm SRCS softmax_compute_test.cc DEPS softmax_compute_arm)
lite_cc_test(test_conv_compute_arm SRCS conv_compute_test.cc DEPS conv_compute_arm)
lite_cc_test(test_batch_norm_compute_arm SRCS batch_norm_compute_test.cc DEPS batch_norm_compute_arm)
lite_cc_test(test_elementwise_compute_arm SRCS elementwise_compute_test.cc DEPS elementwise_compute_arm)
lite_cc_test(test_lrn_compute_arm SRCS lrn_compute_test.cc DEPS lrn_compute_arm)
lite_cc_test(test_decode_bboxes_compute_arm SRCS decode_bboxes_compute_test.cc DEPS decode_bboxes_compute_arm)
lite_cc_test(test_multiclass_nms_compute_arm SRCS multiclass_nms_compute_test.cc DEPS multiclass_nms_compute_arm)
lite_cc_test(test_pool_compute_arm SRCS pool_compute_test.cc DEPS pool_compute_arm)
lite_cc_test(test_mul_compute_arm SRCS mul_compute_test.cc DEPS mul_compute_arm)
lite_cc_test(test_split_compute_arm SRCS split_compute_test.cc DEPS split_compute_arm)
lite_cc_test(test_concat_compute_arm SRCS concat_compute_test.cc DEPS concat_compute_arm)
lite_cc_test(test_dropout_compute_arm SRCS dropout_compute_test.cc DEPS dropout_compute_arm)
lite_cc_test(test_transpose_compute_arm SRCS transpose_compute_test.cc DEPS transpose_compute_arm)
lite_cc_test(test_argmax_compute_arm SRCS argmax_compute_test.cc DEPS argmax_compute_arm)
lite_cc_test(test_axpy_compute_arm SRCS axpy_compute_test.cc DEPS axpy_compute_arm)
lite_cc_test(test_conv_transpose_compute_arm SRCS conv_transpose_compute_test.cc DEPS conv_transpose_compute_arm)


set(arm_kernels
    fc_compute_arm
    activation_compute_arm
    mul_compute_arm
    scale_compute_arm
    softmax_compute_arm
    conv_compute_arm
    batch_norm_compute_arm
    elementwise_compute_arm
    lrn_compute_arm
    decode_bboxes_compute_arm
    multiclass_nms_compute_arm
    pool_compute_arm
    split_compute_arm
    concat_compute_arm
    pad2d_compute_arm
    prior_box_compute_arm
    density_prior_box_compute_arm
    negative_compute_arm
    crop_compute_arm
    dropout_compute_arm
    transpose_compute_arm
    calib_compute_arm
    argmax_compute_arm
    axpy_compute_arm
    conv_transpose_compute_arm
    gru_unit_compute_arm
    gru_compute_arm
    beam_search_decode_compute_arm
    im2sequence_compute_arm
    sequence_softmax_compute_arm
    norm_compute_arm
    logical_compute_arm
    compare_compute_arm
    topk_compute_arm
    increment_compute_arm
    write_to_array_compute_arm
    read_from_array_compute_arm
    beam_search_compute_arm
    box_coder_compute_arm
    reduce_max_compute_arm
    sequence_expand_compute_arm
    sequence_pool_compute_arm
    )

set(arm_kernels "${arm_kernels}" CACHE INTERNAL "arm kernels")
