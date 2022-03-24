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

#include "operation/multiclass_nms.h"
#include <vector>
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareMulticlassNMS(core::Operation* operation) {
  MULTICLASS_NMS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  std::vector<int> output_shape;
  auto input_dims = input_operands[0]->type.dimensions.count;
  auto input_dims_ptr = input_operands[0]->type.dimensions.data;
  NNADAPTER_CHECK_EQ(input_dims, 3) << "Box dims should be 3D.";
  output_shape.push_back(input_dims_ptr[0] * input_dims_ptr[1]);
  output_shape.push_back(6);
  output_operand->type.dimensions.data[0] = output_shape[0];
  output_operand->type.dimensions.data[1] = output_shape[1];

  // Dynamic shape
  if (input_operands[0]->type.dimensions.dynamic_count != 0) {
    auto input_dims_ptr_dy = input_operands[0]->type.dimensions.dynamic_data;
    std::vector<int> output_shape_1;
    output_shape_1.push_back(input_dims_ptr_dy[0] * input_dims_ptr_dy[1]);
    output_shape_1.push_back(6);
    output_operand->type.dimensions.dynamic_data[0] = output_shape_1[0];
    output_operand->type.dimensions.dynamic_data[1] = output_shape_1[1];
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
