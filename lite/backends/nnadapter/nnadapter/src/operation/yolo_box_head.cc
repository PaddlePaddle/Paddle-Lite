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

#include "operation/yolo_box_head.h"
#include <iostream>
#include "core/types.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"
namespace nnadapter {
namespace operation {

int PrepareYoloBoxHead(core::Operation* operation) {
  YOLO_BOX_HEAD_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape of boxes and scores
  auto& output_type = output_operand->type;
  output_type.dimensions.count = 4;
  int* x_dims = input_operand->type.dimensions.data;

  output_type.dimensions.data[0] = x_dims[0];
  output_type.dimensions.data[1] = x_dims[1];
  output_type.dimensions.data[2] = x_dims[2];
  output_type.dimensions.data[3] = x_dims[3];

  class_num = class_num;
  conf_thresh = conf_thresh;
  downsample_ratio = downsample_ratio;
  clip_bbox = clip_bbox;
  scale_x_y = scale_x_y;
  iou_aware = iou_aware;
  iou_aware_factor = iou_aware_factor;

  output_type.precision = input_operand->type.precision;
  output_type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;

  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
