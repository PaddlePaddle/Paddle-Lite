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

#include "driver/intel_openvino/converter/converter.h"
#include "operation/arg_min_max.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertArgMinMax(Converter* converter, core::Operation* operation) {
  ARG_MIN_MAX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto k_tensor = converter->AddConstantTensor<int64_t>(1);
  auto axis_tensor = converter->AddConstantTensor<uint64_t>(axis);
  auto topk_op =
      std::make_shared<default_opset::TopK>(*input_tensor,
                                            *k_tensor,
                                            axis,
                                            "max",
                                            "index",
                                            ConvertToOVElementType(dtype));
  auto squeeze_op = std::make_shared<default_opset::Squeeze>(topk_op->output(1),
                                                             *axis_tensor);
  auto output_tensor = MAP_OUTPUT(output_operand, squeeze_op, 0);
  if (keepdim) {
    auto axes_tensor =
        converter->AddConstantTensor<int32_t>(std::vector<int32_t>({axis}));
    auto unsqueeze_op = std::make_shared<default_opset::Unsqueeze>(
        squeeze_op->output(0), *axes_tensor);
    MAP_OUTPUT(output_operand, unsqueeze_op, 0);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
