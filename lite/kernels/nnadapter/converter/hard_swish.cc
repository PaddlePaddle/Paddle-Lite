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

#include <cmath>
#include "lite/kernels/nnadapter/converter/converter.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertHardSwish(Converter* converter, OpInfo* op, Scope* scope) {
  // Extract op attributes
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto out_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
  }
  auto offset = op->GetAttr<float>("offset");
  auto threshold = op->GetAttr<float>("threshold");
  auto scale = op->GetAttr<float>("scale");
  float mul_factor = threshold / scale;
  // Check quantization mode
  bool is_quant_mode = IsValidSymmPerLayerQuantParams(out_scales);
  if (is_quant_mode) {
    CHECK(IsValidSymmPerLayerQuantParams(x_scales))
        << "Missing the quant params '" << x_scale_name << "' for the input '"
        << x_name << "'";
  }

  // Convert to NNAdapter operands and operation
  // output = MUL(HARD_SWISH(input, alpha = 1 / threshold, beta = offset /
  // threshold), threshold / scale);
  // Input operand
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);
  CHECK(input_operand);
  auto input_type = converter->GetOperandType(input_operand);
  if (is_quant_mode) {
    CHECK(IsNNInt8SymmPerLayerQuantType(*input_type));
    std::vector<float> quant_scales;
    CHECK(GetNNSymmQuantParams(*input_type, &quant_scales));
    CHECK(IsSameSymmQuantParams(x_scales, quant_scales));
  }
  // Alpha operand
  auto alpha_operand = converter->AddConstantOperand(1.0f / threshold);
  // Beta operand
  auto beta_operand = converter->AddConstantOperand(offset / threshold);
  // Output operand
  auto output_operand = converter->AddOutputOperand(out_name, out_scales);
  // Add HARD_SWISH operation
  converter->AddOperation(NNADAPTER_HARD_SWISH,
                          {input_operand, alpha_operand, beta_operand},
                          {output_operand});
  // Add MUL operation if mul_factor != 1.0
  if (fabs(mul_factor - 1.0f) >= 1e-5f) {
    auto mul_factor_operand = converter->AddConstantOperand(mul_factor);
    auto fuse_code_operand = converter->AddConstantOperand(
        static_cast<int32_t>(NNADAPTER_FUSED_NONE));
    auto mul_output_operand = converter->AddOutputOperand(out_name, out_scales);
    converter->AddOperation(
        NNADAPTER_MUL,
        {output_operand, mul_factor_operand, fuse_code_operand},
        {mul_output_operand});
  }
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
