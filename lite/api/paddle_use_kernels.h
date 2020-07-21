#pragma once
#include "paddle_lite_factory_helper.h"

USE_LITE_KERNEL(crop, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fill_constant_batch_size_like, kARM, kAny, kNCHW, def);
USE_LITE_KERNEL(feed, kHost, kAny, kAny, def);
USE_LITE_KERNEL(squeeze, kHost, kAny, kAny, def);
USE_LITE_KERNEL(squeeze2, kHost, kAny, kAny, def);
USE_LITE_KERNEL(equal, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(equal, kHost, kInt32, kAny, def);
USE_LITE_KERNEL(not_equal, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(less_than, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(less_than, kHost, kInt32, kAny, def);
USE_LITE_KERNEL(less_than, kHost, kInt64, kAny, def);
USE_LITE_KERNEL(less_equal, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(greater_than, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(greater_equal, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(top_k, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(mul, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(unsqueeze, kHost, kAny, kAny, def);
USE_LITE_KERNEL(unsqueeze2, kHost, kAny, kAny, def);
USE_LITE_KERNEL(sequence_softmax, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(mean, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(reshape, kHost, kAny, kAny, def);
USE_LITE_KERNEL(reshape2, kHost, kAny, kAny, def);
USE_LITE_KERNEL(flatten, kHost, kAny, kAny, def);
USE_LITE_KERNEL(flatten2, kHost, kAny, kAny, def);
USE_LITE_KERNEL(box_clip, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(expand, kHost, kFloat, kAny, def);
USE_LITE_KERNEL(conv2d_transpose, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(arg_max, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(assign, kHost, kAny, kAny, def);
USE_LITE_KERNEL(write_to_array, kHost, kAny, kAny, def);
USE_LITE_KERNEL(lstm, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(increment, kARM, kAny, kNCHW, def);
USE_LITE_KERNEL(reduce_prod, kARM, kInt32, kNCHW, def);
USE_LITE_KERNEL(reduce_prod, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(norm, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(read_from_array, kHost, kAny, kAny, def);
USE_LITE_KERNEL(gather, kARM, kAny, kNCHW, def);
USE_LITE_KERNEL(reduce_mean, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(gru, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(bilinear_interp, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(nearest_interp, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(split_lod_tensor, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_add, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_add_activation, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_sub, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_sub_activation, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_mul, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_mul, kARM, kInt32, kNCHW, def);
USE_LITE_KERNEL(elementwise_mul, kARM, kInt64, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_mul_activation, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_max, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_max_activation, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_div, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_div_activation, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_expand, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(collect_fpn_proposals, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(concat, kARM, kAny, kNCHW, def);
USE_LITE_KERNEL(cast, kARM, kAny, kNCHW, def);
USE_LITE_KERNEL(lookup_table, kARM, kAny, kNCHW, def);
USE_LITE_KERNEL(lookup_table_v2, kARM, kAny, kNCHW, def);
USE_LITE_KERNEL(split, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(generate_proposals, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(slice, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(slice, kARM, kInt32, kNCHW, def);
USE_LITE_KERNEL(gru_unit, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(dropout, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(lod_reset, kARM, kAny, kNCHW, def);
USE_LITE_KERNEL(range, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(pad2d, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(stack, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(while, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(relu, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(leaky_relu, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(relu_clipped, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(prelu, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(sigmoid, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(tanh, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(swish, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(relu6, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(log, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(exp, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(floor, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(hard_sigmoid, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(rsqrt, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(square, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(hard_swish, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(reciprocal, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(abs, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(lookup_table_dequant, kARM, kAny, kNCHW, def);
USE_LITE_KERNEL(deformconv2d, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(box_coder, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(batch_norm, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(roi_align, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fetch, kHost, kAny, kAny, def);
USE_LITE_KERNEL(sequence_pool, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(anchor_generator, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fc, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fc, kARM, kInt8, kNCHW, int8out);
USE_LITE_KERNEL(fc, kARM, kInt8, kNCHW, fp32out);
USE_LITE_KERNEL(affine_grid, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(conv2d, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(depthwise_conv2d, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(conv2d, kARM, kInt8, kNCHW, int8_out);
USE_LITE_KERNEL(conv2d, kARM, kInt8, kNCHW, fp32_out);
USE_LITE_KERNEL(depthwise_conv2d, kARM, kInt8, kNCHW, int8_out);
USE_LITE_KERNEL(depthwise_conv2d, kARM, kInt8, kNCHW, fp32_out);
USE_LITE_KERNEL(scale, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(scale, kARM, kInt32, kNCHW, def);
USE_LITE_KERNEL(matmul, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(assign_value, kARM, kAny, kNCHW, def);
USE_LITE_KERNEL(ctc_align, kHost, kInt64, kNCHW, def);
USE_LITE_KERNEL(ctc_align, kHost, kInt32, kNCHW, def);
USE_LITE_KERNEL(conditional_block, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(beam_search, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(crf_decoding, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(transpose, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(transpose2, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(multiclass_nms, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(multiclass_nms2, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(shape, kHost, kAny, kAny, def);
USE_LITE_KERNEL(instance_norm, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(calib, kARM, kInt8, kNCHW, fp32_to_int8);
USE_LITE_KERNEL(calib, kARM, kInt8, kNCHW, int8_to_fp32);
USE_LITE_KERNEL(calib, kARM, kInt8, kNHWC, fp32_to_int8);
USE_LITE_KERNEL(calib, kARM, kInt8, kNHWC, int8_to_fp32);
USE_LITE_KERNEL(calib_once, kARM, kInt8, kNCHW, fp32_to_int8);
USE_LITE_KERNEL(calib_once, kARM, kInt8, kNCHW, int8_to_fp32);
USE_LITE_KERNEL(calib_once, kARM, kInt8, kNHWC, fp32_to_int8);
USE_LITE_KERNEL(calib_once, kARM, kInt8, kNHWC, int8_to_fp32);
USE_LITE_KERNEL(is_empty, kHost, kAny, kAny, def);
USE_LITE_KERNEL(sequence_conv, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(lrn, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(axpy, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(shuffle_channel, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(pool2d, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(logical_xor, kHost, kAny, kAny, def);
USE_LITE_KERNEL(logical_and, kHost, kAny, kAny, def);
USE_LITE_KERNEL(logical_or, kHost, kAny, kAny, def);
USE_LITE_KERNEL(logical_not, kHost, kAny, kAny, def);
USE_LITE_KERNEL(beam_search_decode, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(density_prior_box, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(affine_channel, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(negative, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(prior_box, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(im2sequence, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(softmax, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(yolo_box, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(decode_bboxes, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(reduce_max, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(layer_norm, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(distribute_fpn_proposals, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(layout, kARM, kFloat, kNCHW, nchw2nhwc);
USE_LITE_KERNEL(layout, kARM, kFloat, kNCHW, nhwc2nchw);
USE_LITE_KERNEL(layout, kARM, kInt8, kNCHW, int8_nchw2nhwc);
USE_LITE_KERNEL(layout, kARM, kInt8, kNCHW, int8_nhwc2nchw);
USE_LITE_KERNEL(layout_once, kARM, kFloat, kNCHW, nchw2nhwc);
USE_LITE_KERNEL(layout_once, kARM, kFloat, kNCHW, nhwc2nchw);
USE_LITE_KERNEL(layout_once, kARM, kInt8, kNCHW, int8_nchw2nhwc);
USE_LITE_KERNEL(layout_once, kARM, kInt8, kNCHW, int8_nhwc2nchw);
USE_LITE_KERNEL(power, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(merge_lod_tensor, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fill_constant, kARM, kAny, kNCHW, def);
USE_LITE_KERNEL(grid_sampler, kARM, kFloat, kNCHW, def);
