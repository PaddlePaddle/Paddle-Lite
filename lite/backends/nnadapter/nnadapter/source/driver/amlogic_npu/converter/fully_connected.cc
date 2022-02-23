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

#include "operation/fully_connected.h"
#include "driver/amlogic_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace amlogic_npu {

int ConvertFullyConnected(Converter* converter, core::Operation* operation) {
  FULLY_CONNECTED_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to amlnpu tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto weight_tensor = converter->ConvertOperand(weight_operand);
  auto bias_tensor = converter->ConvertOperand(bias_operand);
  auto output_tensor = converter->ConvertOperand(output_operand);
  aml::nn::FCAttr attr;
  attr.weights = num_units;
  // fuse RELU ?
  if (fuse_code == NNADAPTER_FUSED_NONE) {
    attr.has_relu = false;
  } else if (fuse_code == NNADAPTER_FUSED_RELU) {
    attr.has_relu = true;
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                         << ") is found.";
  }
  std::vector<std::shared_ptr<aml::nn::Tensor>> input_tensors = {
      input_tensor, weight_tensor, bias_tensor};
  std::vector<std::shared_ptr<aml::nn::Tensor>> output_tensors = {
      output_tensor};
  converter->AddOperator(
      aml::nn::OperatorType::FULLCONNECT, input_tensors, output_tensors, &attr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace amlogic_npu
}  // namespace nnadapter
