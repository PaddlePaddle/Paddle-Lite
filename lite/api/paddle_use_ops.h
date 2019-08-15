// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// ATTENTION This can only include in a .cc file.

#include "paddle_lite_factory_helper.h"  // NOLINT

USE_LITE_OP(mul);
USE_LITE_OP(fc);
USE_LITE_OP(relu);
USE_LITE_OP(relu6);
USE_LITE_OP(scale);
USE_LITE_OP(feed);
USE_LITE_OP(lrn);
USE_LITE_OP(decode_bboxes);
USE_LITE_OP(box_coder);
USE_LITE_OP(fetch);
USE_LITE_OP(io_copy);
USE_LITE_OP(io_copy_once);
USE_LITE_OP(elementwise_add)
USE_LITE_OP(elementwise_sub)
USE_LITE_OP(elementwise_mul)
USE_LITE_OP(elementwise_max)
USE_LITE_OP(fusion_elementwise_add_activation)
USE_LITE_OP(fusion_elementwise_mul_activation)
USE_LITE_OP(fusion_elementwise_max_activation)
USE_LITE_OP(square)
USE_LITE_OP(softmax)
USE_LITE_OP(dropout)
USE_LITE_OP(concat)
USE_LITE_OP(conv2d)
USE_LITE_OP(depthwise_conv2d)
USE_LITE_OP(pool2d)
USE_LITE_OP(batch_norm)
USE_LITE_OP(fusion_elementwise_sub_activation)
USE_LITE_OP(transpose)
USE_LITE_OP(transpose2)
USE_LITE_OP(argmax)
USE_LITE_OP(axpy)
USE_LITE_OP(leaky_relu)
USE_LITE_OP(relu_clipped)
USE_LITE_OP(prelu)
USE_LITE_OP(sigmoid)
USE_LITE_OP(tanh)
USE_LITE_OP(swish)
USE_LITE_OP(log)
USE_LITE_OP(exp)
USE_LITE_OP(conv2d_transpose)
USE_LITE_OP(negative)
USE_LITE_OP(pad2d)
USE_LITE_OP(power)
USE_LITE_OP(shuffle_channel)
USE_LITE_OP(yolo_box)
USE_LITE_OP(bilinear_interp)
USE_LITE_OP(nearest_interp)

USE_LITE_OP(crop)
USE_LITE_OP(prior_box)
USE_LITE_OP(density_prior_box)
USE_LITE_OP(reshape)
USE_LITE_OP(reshape2)
USE_LITE_OP(split)
USE_LITE_OP(fake_quantize_moving_average_abs_max);
USE_LITE_OP(fake_dequantize_max_abs);
USE_LITE_OP(calib);
USE_LITE_OP(calib_once);
USE_LITE_OP(norm);
USE_LITE_OP(layout);
USE_LITE_OP(layout_once);
USE_LITE_OP(im2sequence);
USE_LITE_OP(sequence_softmax);
USE_LITE_OP(logical_xor);
USE_LITE_OP(logical_and);
USE_LITE_OP(less_than);
USE_LITE_OP(top_k);
USE_LITE_OP(increment);
USE_LITE_OP(write_to_array);
USE_LITE_OP(read_from_array);
USE_LITE_OP(gru_unit)
USE_LITE_OP(gru)
USE_LITE_OP(beam_search_decode)
USE_LITE_OP(beam_search)
USE_LITE_OP(fill_constant)
USE_LITE_OP(while)
USE_LITE_OP(lod_reset)
USE_LITE_OP(lookup_table)
USE_LITE_OP(multiclass_nms)
USE_LITE_OP(graph_op)
USE_LITE_OP(sequence_expand)
USE_LITE_OP(sequence_pool)
USE_LITE_OP(reduce_max)
USE_LITE_OP(is_empty)
USE_LITE_OP(shape)
USE_LITE_OP(slice)
USE_LITE_OP(cast)
USE_LITE_OP(squeeze)   // for x2paddle
USE_LITE_OP(squeeze2)  // for x2paddle
USE_LITE_OP(expand)    // for x2paddle
