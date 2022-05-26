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

#include "operation/softplus.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertSoftplus(Converter* converter, core::Operation* operation) {
  SOFTPLUS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  const float EPSINON = 1e-6;
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  // Only support beta==1.0 && threshold==20.0 for now.
  NNADAPTER_CHECK(std::fabs(beta - 1) <= EPSINON &&
                  std::fabs(threshold - 20) <= EPSINON);
  auto softplus_op = std::make_shared<default_opset::SoftPlus>(*input_tensor);
  MAP_OUTPUT(output_operand, softplus_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
