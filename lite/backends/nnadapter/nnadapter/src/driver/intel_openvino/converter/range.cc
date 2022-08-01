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

#include "operation/range.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertRange(Converter* converter, core::Operation* operation) {
  RANGE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto start_tensor = converter->GetMappedTensor(start_operand);
  if (!start_tensor) {
    start_tensor = converter->ConvertOperand(start_operand);
  }
  auto limit_tensor = converter->GetMappedTensor(limit_operand);
  if (!limit_tensor) {
    limit_tensor = converter->ConvertOperand(limit_operand);
  }
  auto delta_tensor = converter->GetMappedTensor(delta_operand);
  if (!delta_tensor) {
    delta_tensor = converter->ConvertOperand(delta_operand);
  }
  auto axis_tensor = converter->AddConstantTensor<int64_t>(0);
  auto start_scalar =
      std::make_shared<default_opset::Squeeze>(*start_tensor, *axis_tensor);
  auto stop_scalar =
      std::make_shared<default_opset::Squeeze>(*limit_tensor, *axis_tensor);
  auto step_scalar =
      std::make_shared<default_opset::Squeeze>(*delta_tensor, *axis_tensor);
  auto range_op =
      std::make_shared<default_opset::Range>(start_scalar->output(0),
                                             stop_scalar->output(0),
                                             step_scalar->output(0),
                                             start_tensor->get_element_type());
  MAP_OUTPUT(output_operand, range_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
