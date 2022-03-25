// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/yolo_box.h"
#include <iostream>
#include "core/types.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"
namespace nnadapter {
namespace operation {

int PrepareYoloBox(core::Operation* operation) {
  YOLO_BOX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape of boxes and scores
  auto& boxes_type = boxes_operand->type;
  auto& scores_type = scores_operand->type;
  boxes_type.dimensions.count = 3;
  scores_type.dimensions.count = 3;
  NNADAPTER_CHECK(IsTemporaryShapeOperand(input_operand));
  NNADAPTER_CHECK(IsTemporaryShapeOperand(imgsize_operand));
  int* x_dims = input_operand->type.dimensions.data;
  int box_num = anchors.size() / 2 * x_dims[2] * x_dims[3];
  boxes_type.dimensions.data[0] = x_dims[0];
  boxes_type.dimensions.data[1] = box_num;
  boxes_type.dimensions.data[2] = 4;
  scores_type.dimensions.data[0] = x_dims[0];
  scores_type.dimensions.data[1] = box_num;
  scores_type.dimensions.data[2] = class_num;

  conf_thresh = conf_thresh;
  downsample_ratio = downsample_ratio;
  clip_bbox = clip_bbox;
  scale_x_y = scale_x_y;
  iou_aware = iou_aware;
  iou_aware_factor = iou_aware_factor;
  CopyOperandTypeWithPrecision(&boxes_type, input_operand->type);
  CopyOperandTypeWithPrecision(&scores_type, input_operand->type);
  NNADAPTER_VLOG(5) << "output: " << OperandToString(boxes_operand);
  NNADAPTER_VLOG(5) << "output: " << OperandToString(scores_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
