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
   * yolo_box_3d + nms fuser introduction:
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32 tensor of shape [N, C, H, W], its
   * dimension(C) stores box locations, confidence score and classification
   * one-hot keys of each anchor box.
   * * 1: input1, a NNADAPTER_FLOAT32 tensor of shape [N, C, H, W], its
   * dimension(C) stores box locations, confidence score and classification
   * one-hot keys of each anchor box.
   * * 2: input2, a NNADAPTER_FLOAT32 tensor of shape [N, C, H, W], its
   * dimension(C) stores box locations, confidence score and classification
   * one-hot keys of each anchor box.
   * * 3: imgsize, a NNADAPTER_INT32 tensor of shape [N, 2], holds height and
   * width of each input image used for resizing output box in input image
   * scale.
   * * 4: anchors0, a vector of NNADAPTER_INT32 scalar, the anchor width and
   *      height of box0, it will be parsed pair by pair.
   * * 5: anchors1, a vector of NNADAPTER_INT32 scalar, the anchor width and
   *      height of box1, it will be parsed pair by pair.
   * * 6: anchors2, a vector of NNADAPTER_INT32 scalar, the anchor width and
   *      height of box2, it will be parsed pair by pair.
   * * 7: class_num, a NNADAPTER_INT32 scalar, number of classes to predict.
   * * 8: conf_thresh, a NNADAPTER_FLOAT32 scalar, the confidence scores
   * * 9: downsample_ratio0, a NNADAPTER_INT32 scalar, down-sampling rate from
   *       network input to this operation input of box0.
   * * 10: downsample_ratio1, a NNADAPTER_INT32 scalar, down-sampling rate from
   *       network input to this operation inputof box1.
   * * 11: downsample_ratio2, a NNADAPTER_INT32 scalar, down-sampling rate from
   *       network input to this operation input of box2.
   * * 12: scale_x_y, a NNADAPTER_FLOAT32 scalar, scale the center point of
   *       decoded bounding box, default 1.0.
   * * 13: background_label, a NNADAPTER_INT32 tensor with shape [1], the index
   * of background label.
   * If set to 0, the background label will be ignored.
   * If set to -1, then all categories will be considered.
   * * 14: score_threshold, a NNADAPTER_FLOAT32 tensor with shape [1], threshold
   * to filter out bounding boxes with low confidence score.
   * * 15: nms_top_k, a NNADAPTER_INT32 tensor with shape [1], maximum number of
   * detections to be kept according to the confidences after the filtering
   * detections based on score_threshold.
   * * 16: nms_threshold, a NNADAPTER_FLOAT32 tensor with shape [1], the
   * parameter for NMS.
   * * 17: nms_eta, a NNADAPTER_FLOAT32 tensor with shape [1], the parameter for
   * adaptive NMS.
   * * 18: keep_top_k, a NNADAPTER_INT32 tensor with shape [1], number of total
   * bboxes to be kept per image after NMS step.
   * "-1" means keeping all bboxes after NMS step.
   * * 19: normalized, a NNADAPTER_BOOL8 tensor with shape [1], whether
   * detections are normalized.
   *
   * Outputs:
   * * 0: out_box, a tensor with the same type as bboxes, with shape [No, 6].
   * "No" is the number of all RoIs. Each row has 6 values: [label, confidence,
   * xmin, ymin, xmax, ymax]
   * * 1: out_rois_num, a NNADAPTER_INT32 tensor of shape [B], B is the number
   * of images.
   * The number of NMS RoIs in each image.
   * * 2: out_index, a NNADAPTER_INT32 tensor with shape [No] represents
   * the index
   * of selected bbox.
   * The out_index is the absolute index cross batches.
   * * 3: location, a 3-D NNADAPTER_FLOAT32 tensor of shape [No, 3].
   * * 4: dim, a 3-D NNADAPTER_FLOAT32 tensor of shape [No, 3].
   * * 5: alpha, a 3-D NNADAPTER_FLOAT32 tensor of shape [No, 2].
   */
  NNADAPTER_CUSTOM_YOLO_BOX_3D_NMS_FUSER = -1000,
};  // Custom operations type

}  // namespace nnadapter
