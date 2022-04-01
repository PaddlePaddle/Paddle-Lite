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

#include "operation/multiclass_nms.h"
#include <vector>
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

bool ValidateMulticlassNMSd(const core::Operation* operation) { return false; }

int PrepareMulticlassNMS(core::Operation* operation) {
  MULTICLASS_NMS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  std::vector<int> output_shape;
  auto input_dims = input_operands[0]->type.dimensions.count;
  auto input_dims_ptr = input_operands[0]->type.dimensions.data;
  NNADAPTER_CHECK_EQ(input_dims, 3) << "Box dims should be 3D.";
  auto N = input_dims_ptr[0];
  auto M = input_dims_ptr[1];
  auto size = N * M;
  output_shape.push_back(size);
  output_shape.push_back(6);
  output_box_operand->type.dimensions.data[0] = output_shape[0];
  output_box_operand->type.dimensions.data[1] = output_shape[1];

  // Dynamic shape
  if (input_operands[0]->type.dimensions.dynamic_count != 0) {
    for (uint32_t i = 0; i < input_operands[0]->type.dimensions.dynamic_count;
         i++) {
      std::vector<int> output_shape(input_count, 0);
      for (size_t j = 0; j < input_count; j++) {
        output_shape[i] = input_operands[i]->type.dimensions.dynamic_data[i][0];
      }
      for (auto output_operand : output_operands) {
        for (int j = 0; j < input_count; j++) {
          output_operand->type.dimensions.dynamic_data[i][j] = output_shape[j];
        }
      }
    }
  }
  return NNADAPTER_NO_ERROR;
}

int ExecuteeMulticlassNMS(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
