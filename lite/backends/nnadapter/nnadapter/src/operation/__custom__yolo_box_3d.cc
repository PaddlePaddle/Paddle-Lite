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

#include "operation/__custom__yolo_box_3d.h"

#include "core/types.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateCustomYoloBox3d(
    const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareCustomYoloBox3d(core::Operation* operation) {
  CUSTOM_YOLO_BOX_3D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape of boxes and scores
  auto& boxes_type = boxes_operand->type;
  auto& scores_type = scores_operand->type;
  auto& location_type = location_operand->type;
  auto& dim_type = dim_operand->type;
  auto& alpha_type = alpha_operand->type;
  boxes_type.dimensions.count = 3;
  scores_type.dimensions.count = 3;
  location_type.dimensions.count = 3;
  dim_type.dimensions.count = 3;
  alpha_type.dimensions.count = 3;

  auto infer_output_shape = [&](int* x_dims,
                                int32_t* boxes_dims,
                                int32_t* scores_dims,
                                int32_t* location_dims,
                                int32_t* dim_dims,
                                int32_t* alpha_dims) {
    int box_num = anchors.size() / 2 * x_dims[2] * x_dims[3];
    boxes_dims[0] = x_dims[0];
    boxes_dims[1] = box_num;
    boxes_dims[2] = 4;
    scores_dims[0] = x_dims[0];
    scores_dims[1] = box_num;
    scores_dims[2] = class_num;
    location_dims[0] = x_dims[0];
    location_dims[1] = box_num;
    location_dims[2] = 3;
    dim_dims[0] = x_dims[0];
    dim_dims[1] = box_num;
    dim_dims[2] = 3;
    alpha_dims[0] = x_dims[0];
    alpha_dims[1] = box_num;
    alpha_dims[2] = 2;
  };

  infer_output_shape(input_operand->type.dimensions.data,
                     boxes_operand->type.dimensions.data,
                     scores_operand->type.dimensions.data,
                     location_operand->type.dimensions.data,
                     dim_operand->type.dimensions.data,
                     alpha_operand->type.dimensions.data);
  for (uint32_t i = 0; i < input_operand->type.dimensions.dynamic_count; i++) {
    infer_output_shape(input_operand->type.dimensions.dynamic_data[i],
                       boxes_operand->type.dimensions.dynamic_data[i],
                       scores_operand->type.dimensions.dynamic_data[i],
                       location_operand->type.dimensions.dynamic_data[i],
                       dim_operand->type.dimensions.dynamic_data[i],
                       alpha_operand->type.dimensions.dynamic_data[i]);
  }
  CopyOperandTypeWithPrecision(&boxes_type, input_operand->type);
  CopyOperandTypeWithPrecision(&scores_type, input_operand->type);
  CopyOperandTypeWithPrecision(&location_type, input_operand->type);
  CopyOperandTypeWithPrecision(&dim_type, input_operand->type);
  CopyOperandTypeWithPrecision(&alpha_type, input_operand->type);
  NNADAPTER_VLOG(5) << "boxes: " << OperandToString(boxes_operand);
  NNADAPTER_VLOG(5) << "scores: " << OperandToString(scores_operand);
  NNADAPTER_VLOG(5) << "location: " << OperandToString(location_operand);
  NNADAPTER_VLOG(5) << "dim: " << OperandToString(dim_operand);
  NNADAPTER_VLOG(5) << "alpha: " << OperandToString(alpha_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteCustomYoloBox3d(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
