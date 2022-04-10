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
   * * 0: input0, a NNADAPTER_FLOAT32 tensor. a 4-D tensor with shape of [N, C,
   *      H, W]. The dimension(C) stores "box locations, confidence score and
   *      classification one-hot keys of each anchor box. Generally, X should be
   * the
   *      output of YOLOv3 network.
   * * 1: anchors, a vector of NNADAPTER_INT32 scalar, the anchor width and
   *      height, it will be parsed pair by pair.
   * * 2: class_num, a NNADAPTER_INT32 scalar, number of classes to predict.
   * * 3: conf_thresh, a NNADAPTER_FLOAT32 scalar, the confidence scores
   *      threshold of detection boxes, boxes with confidence scores under
   * threshold
   *      should be ignored.
   * * 4: downsample_ratio, a NNADAPTER_INT32 scalar, down-sampling rate from
   *      network input to this operation input.
   * * 5: clip_bbox, a NNADAPTER_BOOL8 scalar, whether clip output bonding box
   *      in input(imgsize), default true.
   * * 6: scale_x_y, a NNADAPTER_FLOAT32 scalar, scale the center point of
   *      decoded bounding box, default 1.0.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_FP32 tensor,  a 4-D tensor with shape of [N, C,
   *      H, W]. The dimension(C) stores "box locations, confidence score and
   *      classification one-hot keys of each anchor box. Generally, X should be
   * the
   *      output of YOLOv3 network.
   */
  NNADAPTER_YOLO_BOX_HEAD,

  /**
   * yolo_box_parser introduction:
   * it finish box parser and filter box according to score_threshold
   *
   * Inputs:
   * * 0: box0, a NNADAPTER_FLOAT32 tensor. a 4-D tensor with shape of [N, C,
   *      H, W]. The dimension(C) stores "box locations, confidence score and
   *      classification one-hot keys of each anchor box. Generally, X should be
   * the
   *      output of YOLOv3 network.
   * * 1: box1, a NNADAPTER_FLOAT32 tensor. a 4-D tensor with shape of [N, C,
   *      H, W]. The dimension(C) stores "box locations, confidence score and
   *      classification one-hot keys of each anchor box. Generally, X should be
   * the
   *      output of YOLOv3 network.
   * * 2: box2, a NNADAPTER_FLOAT32 tensor. a 4-D tensor with shape of [N, C,
   *      H, W]. The dimension(C) stores "box locations, confidence score and
   *      classification one-hot keys of each anchor box. Generally, X should be
   * the
   *      output of YOLOv3 network.
   * * 3: image_shape, a NNADAPTER_INT32 tensor. a 2-D tensor with shape of
   *      [N, 2]. This tensor holds height and width of each input image.
   * * 4: image_scale, a NNADAPTER_INT32 tensor. a 2-D tensor with shape of
   *      [N, 2]. This tensor holds height and width of each input image used
   * for
   *      resizing output box in input image scale.
   * * 5: anchors0, a vector of NNADAPTER_INT32 scalar, the anchor width and
   *      height of box0, it will be parsed pair by pair.
   * * 6: anchors1, a vector of NNADAPTER_INT32 scalar, the anchor width and
   *      height of box1, it will be parsed pair by pair.
   * * 7: anchors2, a vector of NNADAPTER_INT32 scalar, the anchor width and
   *      height of box2, it will be parsed pair by pair.
   * * 8: class_num, a NNADAPTER_INT32 scalar, number of classes to predict.
   * * 9: conf_thresh, a NNADAPTER_FLOAT32 scalar, the confidence scores
   * * 10: downsample_ratio0, a NNADAPTER_INT32 scalar, down-sampling rate from
   *       network input to this operation input of box0.
   * * 11: downsample_ratio1, a NNADAPTER_INT32 scalar, down-sampling rate from
   *       network input to this operation inputof box1.
   * * 12: downsample_ratio2, a NNADAPTER_INT32 scalar, down-sampling rate from
   *       network input to this operation input of box2.
   * * 13: clip_bbox, a NNADAPTER_BOOL8 scalar, whether clip output bonding box
   *       in input(imgsize), default true.
   * * 14: scale_x_y, a NNADAPTER_FLOAT32 scalar, scale the center point of
   *       decoded bounding box, default 1.0.
   * * 15: nms_threshold, a NNADAPTER_FLOAT32 tensor with shape [1], the
   *       parameter for NMS.
   *
   * Outputs:
   * * 0: output, a tensor with the same type as bboxes, with shape [No, 6].
   *      "No" is the number of all RoIs. Each row has 6 values: [label,
   * confidence,
   *      xmin, ymin, xmax, ymax]
   * * 1: out_rois_num, a NNADAPTER_INT32 tensor with shape [B], B is the number
   *      of images. The number of NMS RoIs in each image.
   */
  NNADAPTER_YOLO_BOX_POST,
};  // Custom operations type

}  // namespace nnadapter
