// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

namespace nnadapter {

enum {
  /**
   * Custom softmax to compute on host place
   * Computes the normalized exponential values for the input tensor
   * element-wise.
   * The output is calculated using this formula:
   *     output = exp(input) / reduce_sum(exp(input), axis=axis, keepdims=true)
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, a NNADAPTER_INT32 scalar. Defaults to 1. It represents the
   * dimension along which softmax will be performed. It should be in range [-R,
   * R), where R is the rank of input, negative value works the same way as
   * axis+R.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_NAIVE_SOFTMAX = -1000,

  /**
   * Custom softmax to compute on cuda place
   * Computes the normalized exponential values for the input tensor
   * element-wise.
   * The output is calculated using this formula:
   *     output = exp(input) / reduce_sum(exp(input), axis=axis, keepdims=true)
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, a NNADAPTER_INT32 scalar. Defaults to 1. It represents the
   * dimension along which softmax will be performed. It should be in range [-R,
   * R), where R is the rank of input, negative value works the same way as
   * axis+R.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_SPECIAL_SOFTMAX,
  /**
   * yolo_box_parser introduction:
   * it finish box calculate, sigmoid and exp.
   *
   * Inputs:
   * * 0: x, a NNADAPTER_FP32 tensor, [N, C, H, W].
   * * 1: anchors, a list, [?, 2].
   * * 2: class_num, int.
   * * 3: conf_thresh, float.
   * * 4: downsample_ratio, int, value=[32, 16, 8].
   * * 5: clip_bbox, bool, optional.
   * * 6: scale_x_y, int, optional, default=1.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_FP32 tensor, [N, C, H, W].
   */
  NNADAPTER_YOLO_BOX_HEAD,
  /**
   * yolo_box_parser introduction:
   * it finish box parser and filter box according to score_threshold
   *
   * Inputs:
   * * 0: box0, a NNADAPTER_FP32 tensor, [N, C, H, W].
   * * 1: box1, a NNADAPTER_FP32 tensor, [N, C, H, W].
   * * 2: box2, a NNADAPTER_FP32 tensor, [N, C, H, W].
   * * 3: image_shape, a NNADAPTER_FP32 tensor, [N, 2].
   * * 4: image_scale, a NNADAPTER_FP32 tensor, [N, 2].
   * * 5: anchors0, a list, [?, 2].
   * * 6: anchors1, a list, [?, 2].
   * * 7: anchors2, a list, [?, 2].
   * * 8: class_num, int.
   * * 9: conf_thresh, float.
   * * 10: downsample_ratio0, int.
   * * 11: downsample_ratio1, int.
   * * 12: downsample_ratio2, int.
   * * 13: clip_bbox, bool, optional.
   * * 14: scale_x_y, int, optional, default=1.
   *
   * Outputs:
   * * 0: boxes_scores, a NNADAPTER_FP32 tensor, [N, M, 4 + class_num].
   */
  NNADAPTER_YOLO_BOX_PARSER,
  /**
   * multiclass_nms introduction:
   * do nms sort process and filter according to nms_threshold
   *
   * Inputs:
   * * 0: box, a NNADAPTER_FP32 tensor, [N, M, 4 + class_num].
   * * 1: background_label, int, default=1.
   * * 2: score_threshold, float.
   * * 3: nms_top_k, int.
   * * 4: nms_threshold, float.
   * * 5: nms_eta, float, default=1.
   * * 6: keep_top_k, int.
   * * 7: normalized, bool,  default=true.
   *
   * Outputs:
   * * 0: box_res, a NNADAPTER_FP32 tensor, [?, 6].
   * * 1: index, a NNADAPTER_INT32 tensor, [?], optional.
   * * 2: NmsRoisNum, a NNADAPTER_INT32 tensor, [?].
   */
  NNADAPTER_YOLO_BOX_NMS,
};  // Custom operations type

}  // namespace nnadapter
