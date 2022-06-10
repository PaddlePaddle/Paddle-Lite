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

#include "operation/layer_normalization.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertLayerNormalization(Converter* converter,
                              core::Operation* operation) {
  LAYER_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto scale_tensor = converter->ConvertOperand(scale_operand);
  auto bias_tensor = converter->ConvertOperand(bias_operand);
  std::vector<int64_t> axes = {};
  auto input_dimension_count = input_operand->type.dimensions.count;
  if (begin_norm_axis < 0) {
    begin_norm_axis += input_dimension_count;
  }
  NNADAPTER_CHECK_LT(begin_norm_axis, input_dimension_count);
  int index = begin_norm_axis;
  for (int i = begin_norm_axis; i < input_dimension_count; i++) {
    axes.push_back(index++);
  }
  auto axes_tensor = converter->AddConstantTensor(axes);
  // Layer normalization
  auto mvn_op =
      std::make_shared<default_opset::MVN>(*input_tensor,
                                           *axes_tensor,
                                           true,
                                           epsilon,
                                           ov::op::MVNEpsMode::INSIDE_SQRT);
  auto multiply_op = std::make_shared<default_opset::Multiply>(
      mvn_op->output(0), *scale_tensor);
  auto add_op = std::make_shared<default_opset::Add>(multiply_op->output(0),
                                                     *bias_tensor);
  MAP_OUTPUT(output_operand, add_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
