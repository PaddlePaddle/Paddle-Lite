// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/transpose.h"
#include "driver/rockchip_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace rockchip_npu {

int ConvertTranspose(Converter* converter, core::Operation* operation) {
  TRANSPOSE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  if (perm_count > 4) {
    NNADAPTER_LOG(FATAL)
        << "Only supports less than 4 dimensions, but perm has " << perm_count;
  }

  // Convert to rknpu tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);
  rk::nn::PermuteAttr attr;
  for (uint32_t i = 0; i < perm_count; i++) {
    attr.perm.push_back(perm_data[i]);
  }
  std::vector<std::shared_ptr<rk::nn::Tensor>> input_tensors = {input_tensor};
  std::vector<std::shared_ptr<rk::nn::Tensor>> output_tensors = {output_tensor};
  converter->AddOperator(
      rk::nn::OperatorType::PERMUTE, input_tensors, output_tensors, &attr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace rockchip_npu
}  // namespace nnadapter
