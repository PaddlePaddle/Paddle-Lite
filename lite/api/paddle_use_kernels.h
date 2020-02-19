#pragma once
#include "paddle_lite_factory_helper.h"

USE_LITE_KERNEL(mul, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(scale, kXPU, kFloat, kNCHW, def);
USE_LITE_KERNEL(squeeze, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(squeeze2, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(search_grnn, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(gru, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(transpose, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(transpose2, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(stack, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(io_copy, kXPU, kAny, kAny, host_to_xpu);
USE_LITE_KERNEL(io_copy, kXPU, kAny, kAny, xpu_to_host);
USE_LITE_KERNEL(io_copy, kXPU, kAny, kAny, x86_to_xpu);
USE_LITE_KERNEL(io_copy, kXPU, kAny, kAny, xpu_to_x86);
USE_LITE_KERNEL(io_copy_once, kXPU, kAny, kAny, host_to_xpu);
USE_LITE_KERNEL(io_copy_once, kXPU, kAny, kAny, xpu_to_host);
USE_LITE_KERNEL(io_copy_once, kXPU, kAny, kAny, x86_to_xpu);
USE_LITE_KERNEL(io_copy_once, kXPU, kAny, kAny, xpu_to_x86);
USE_LITE_KERNEL(reshape, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(reshape2, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(reshape2, kX86, kInt64, kNCHW, def);
USE_LITE_KERNEL(cast, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(reshape, kHost, kAny, kAny, def);
USE_LITE_KERNEL(reshape2, kHost, kAny, kAny, def);
USE_LITE_KERNEL(flatten, kHost, kAny, kAny, def);
USE_LITE_KERNEL(flatten2, kHost, kAny, kAny, def);
USE_LITE_KERNEL(lookup_table, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(lookup_table_v2, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(scale, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(dropout, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(fc, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(var_conv_2d, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(search_seq_fc, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(square, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(relu, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(leaky_relu, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(tanh, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(gelu, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(softsign, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(gather, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(gather, kX86, kFloat, kNCHW, int64_in);
USE_LITE_KERNEL(search_group_padding, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(subgraph, kXPU, kFloat, kNCHW, def);
USE_LITE_KERNEL(match_matrix_tensor, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(pool2d, kXPU, kFloat, kNCHW, def);
USE_LITE_KERNEL(search_attention_padding_mask, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_reverse, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_reverse, kX86, kInt64, kNCHW, def);
USE_LITE_KERNEL(elementwise_add, kXPU, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_sub, kXPU, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_reshape, kX86, kInt64, kNCHW, def);
USE_LITE_KERNEL(mul, kXPU, kFloat, kNCHW, def);
USE_LITE_KERNEL(search_aligned_mat_mul, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_topk_avg_pooling, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(search_seq_depadding, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(softmax, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(search_seq_softmax, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(conv2d, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(depthwise_conv2d, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(softmax, kXPU, kFloat, kNCHW, def);
USE_LITE_KERNEL(feed, kHost, kAny, kAny, def);
USE_LITE_KERNEL(matmul, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(multiclass_nms, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(layer_norm, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_pool, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(concat, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(batch_norm, kXPU, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_arithmetic, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(search_seq_arithmetic, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(conv2d, kXPU, kFloat, kNCHW, def);
USE_LITE_KERNEL(slice, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(reduce_sum, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(batch_norm, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(pool2d, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(fetch, kHost, kAny, kAny, def);
USE_LITE_KERNEL(relu, kXPU, kFloat, kNCHW, def);
USE_LITE_KERNEL(fill_constant_batch_size_like, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(search_fc, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_concat, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(shape, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_expand_as, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_sub, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_add, kX86, kFloat, kNCHW, def);