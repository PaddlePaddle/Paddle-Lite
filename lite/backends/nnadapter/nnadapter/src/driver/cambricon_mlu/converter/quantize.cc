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

#include "operation/quantize.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertQuantize(Converter* converter, core::Operation* operation) {
  QUANTIZE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(is_per_layer_quant)
      << "CambriconMLU currently only support per layer quantize.";

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (input_tensor == nullptr) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto scale_tensor = converter->GetMappedTensor(scale_operand);
  if (scale_tensor == nullptr) {
    scale_tensor = converter->ConvertOperand(scale_operand);
  }
  float scale = scale_data[0];
  magicmind::Range rng = magicmind::UniformQuantParamToRangeWithQuantAlg(
      {scale, 0}, 8, "symmetric");
  auto quantize_node = converter->network()->AddIQuantizeNode(
      input_tensor, rng, true, magicmind::DataType::QINT8, "symmetric");
  NNADAPTER_CHECK(quantize_node) << "Failed to add quantize node.";
  auto output_tensor = quantize_node->GetOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
