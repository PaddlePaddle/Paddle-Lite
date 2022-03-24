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

#include "operation/yolo_box.h"
#include <vector>
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareYoloBox(core::Operation* operation) {
  YOLO_BOX_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Infer the shape and type of output operands
  auto input_dims = input_operands[0]->type.dimensions.count;
  auto input_dims_ptr = input_operands[0]->type.dimensions.data;
  auto anchor_num = anchors.size() / 2;
  auto class_num = *reinterpret_cast<int32_t*>(class_num_operand->buffer);
  NNADAPTER_CHECK_EQ(input_dims, 4) << "Input dims should be 4D.";
  int box_num = input_dims_ptr[2] * input_dims_ptr[3] * anchor_num;
  std::vector<int> output_box_shape;
  output_box_shape.push_back(input_dims_ptr[0]);
  output_box_shape.push_back(box_num);
  output_box_shape.push_back(4);
  std::vector<int> output_score_shape;
  output_score_shape.push_back(input_dims_ptr[0]);
  output_score_shape.push_back(box_num);
  output_score_shape.push_back(class_num);
  for (int i = 0; i < 3; i++) {
    output_box_operand->type.dimensions.data[i] = output_box_shape[i];
    output_score_operand->type.dimensions.data[i] = output_score_shape[i];
  }

  // Dynamic shape
  if (input_operands[0]->type.dimensions.dynamic_count != 0) {
    auto input_dims_ptr_dy = input_operands[0]->type.dimensions.dynamic_data;
    std::vector<int> output_box_shape_dy;
    output_box_shape_dy.push_back(input_dims_ptr_dy[0]);
    output_box_shape_dy.push_back(box_num);
    output_box_shape_dy.push_back(4);
    std::vector<int> output_score_shape;
    output_score_shape_dy.push_back(input_dims_ptr_dy[0]);
    output_score_shape_dy.push_back(box_num);
    output_score_shape_dy.push_back(class_num);
    for (int i = 0; i < 3; i++) {
      output_box_operand->type.dimensions.dynamic_data[i] =
          output_box_shape_dy[i];
      output_score_operand->type.dimensions.dynamic_data[i] =
          output_score_shape_dy[i];
    }
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
