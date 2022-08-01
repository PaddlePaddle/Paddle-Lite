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
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateYoloBox(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareYoloBox(core::Operation* operation) {
  YOLO_BOX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape of boxes and scores
  auto& boxes_type = boxes_operand->type;
  auto& scores_type = scores_operand->type;
  boxes_type.dimensions.count = 3;
  scores_type.dimensions.count = 3;

  auto infer_output_shape = [&](
      int* x_dims, int32_t* boxes_dims, int32_t* scores_dims) {
    int box_num = anchors.size() / 2 * x_dims[2] * x_dims[3];
    boxes_dims[0] = x_dims[0];
    boxes_dims[1] = box_num;
    boxes_dims[2] = 4;
    scores_dims[0] = x_dims[0];
    scores_dims[1] = box_num;
    scores_dims[2] = class_num;
  };

  infer_output_shape(input_operand->type.dimensions.data,
                     boxes_operand->type.dimensions.data,
                     scores_operand->type.dimensions.data);
  for (uint32_t i = 0; i < input_operand->type.dimensions.dynamic_count; i++) {
    infer_output_shape(input_operand->type.dimensions.dynamic_data[i],
                       boxes_operand->type.dimensions.dynamic_data[i],
                       scores_operand->type.dimensions.dynamic_data[i]);
  }
  CopyOperandTypeWithPrecision(&boxes_type, input_operand->type);
  CopyOperandTypeWithPrecision(&scores_type, input_operand->type);
  NNADAPTER_VLOG(5) << "boxes: " << OperandToString(boxes_operand);
  NNADAPTER_VLOG(5) << "scores: " << OperandToString(scores_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteYoloBox(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
