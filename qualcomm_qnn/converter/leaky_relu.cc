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

#include "operation/leaky_relu.h"
#include <memory>
#include "driver/qualcomm_qnn/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace qualcomm_qnn {

int ConvertLeakyRelu(Converter* converter, core::Operation* operation) {
  LEAKY_RELU_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to qnn tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  auto output_tensor = converter->GetMappedTensor(output_operand);
  if (IsAsymmetricQuantType(input_operand->type.precision)) {
    float scale = alpha;
    int32_t zero_point = 0;
    auto alpha_quant_tensor = converter->AddConstantTensor(
        std::vector<int8_t>{1}, {1}, &scale, &zero_point);
    converter->AddNode(
        QNN_OP_PRELU, {input_tensor, alpha_quant_tensor}, {output_tensor});
  } else {
    auto alpha_tensor = converter->GetMappedTensor(input_operands[1]);
    converter->AddNode(
        QNN_OP_PRELU, {input_tensor, alpha_tensor}, {output_tensor});
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
