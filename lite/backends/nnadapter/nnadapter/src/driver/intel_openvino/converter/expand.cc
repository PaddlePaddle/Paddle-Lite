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
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "operation/expand.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertExpand(Converter* converter, core::Operation* operation) {
  EXPAND_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto shape_tensor = converter->GetMappedTensor(shape_operand);
  if (!shape_tensor) {
    shape_tensor = converter->ConvertOperand(shape_operand);
  }
  // Expected shape rank.
  auto shape_expected_rank = std::make_shared<default_opset::ShapeOf>(
      *shape_tensor, GetElementType<int32_t>());
  // Input shape and rank.
  auto input_shape = std::make_shared<default_opset::ShapeOf>(
      *input_tensor, GetElementType<int32_t>());
  auto input_shape_rank = std::make_shared<default_opset::ShapeOf>(
      input_shape->output(0), GetElementType<int32_t>());
  // Rank difference between input shape and expected shape.
  auto rank_diff = std::make_shared<default_opset::Subtract>(
      shape_expected_rank->output(0), input_shape_rank->output(0));
  // Axis index needed to add.
  auto rank_idx = std::make_shared<default_opset::Broadcast>(
      *(converter->AddConstantTensor<int32_t>(std::vector<int32_t>({1}))),
      rank_diff);
  // Add axis.
  auto fixed_input_shape = std::make_shared<default_opset::Concat>(
      TensorVector{rank_idx->output(0), input_shape->output(0)}, 0);
  // If -1 in shape we will copy the orginal value from input.
  auto zero_tensor =
      converter->AddConstantTensor<int32_t>(std::vector<int32_t>({0}));
  auto mask_op =
      std::make_shared<default_opset::Greater>(*shape_tensor, *zero_tensor);
  auto fixed_shape = std::make_shared<default_opset::Select>(
      mask_op->output(0), *shape_tensor, fixed_input_shape->output(0));
  auto broadcast_op = std::make_shared<default_opset::Broadcast>(
      *input_tensor, fixed_shape->output(0));
  MAP_OUTPUT(output_operand, broadcast_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
