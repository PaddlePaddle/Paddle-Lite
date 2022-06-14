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

#include "operation/non_max_suppression.h"
#include <vector>
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

bool ValidateNonMaxSuppression(const core::Operation* operation) {
  return false;
}

int PrepareNonMaxSuppression(core::Operation* operation) {
  NON_MAX_SUPPRESSION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  std::vector<int> output_shape;
  auto input_dims = input_operands[0]->type.dimensions.count;
  NNADAPTER_CHECK_EQ(input_dims, 3) << "Box dims should be 3D.";
  output_shape.push_back(NNADAPTER_UNKNOWN);
  output_shape.push_back(6);
  output_box_operand->type.dimensions.count = 2;
  output_box_operand->type.dimensions.data[0] = output_shape[0];
  output_box_operand->type.dimensions.data[1] = output_shape[1];
  output_nms_rois_num_operand->type.dimensions.count = 1;
  output_nms_rois_num_operand->type.dimensions.data[0] = output_shape[0];
  auto& output_box_type = output_box_operand->type;
  output_box_type.precision = NNADAPTER_FLOAT32;
  output_box_type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  output_nms_rois_num_operand->type.precision = NNADAPTER_INT32;
  output_nms_rois_num_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  if (return_index) {
    output_index_operand->type.dimensions.count = 1;
    output_index_operand->type.dimensions.data[0] = output_shape[0];
    output_index_operand->type.precision = NNADAPTER_INT32;
    output_index_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  }
  return NNADAPTER_NO_ERROR;
}

int ExecuteNonMaxSuppression(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
