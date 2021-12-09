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

#include "lite/kernels/nnadapter/converter/converter.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertCalib(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_name = op->Input("Input").front();
  auto input_operand = converter->AddInputOperand(scope, x_name);

  // Axis operand
  auto axis_operand = converter->AddConstantOperand<int>(1);

  // Scale operand
  CHECK(op->HasAttr("scale"));
  float scale = op->GetAttr<float>("scale");
  auto scale_operand = converter->AddConstantOperand(scale);

  // Zero_point operand
  auto zero_point_operand = converter->AddConstantOperand<int32_t>(0);

  // Output operand
  auto out_name = op->Output("Out").front();
  std::vector<float> out_scales{scale};
  auto output_operand = converter->AddOutputOperand(out_name, out_scales);

  // Quant or dequant operation
  auto input_type = converter->GetOperandType(input_operand);
  switch (input_type->precision) {
    case NNADAPTER_FLOAT32:
    case NNADAPTER_INT32: {
      converter->AddOperation(
          NNADAPTER_QUANTIZE,
          {input_operand, axis_operand, scale_operand, zero_point_operand},
          {output_operand});
      break;
    }
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL: {
      converter->AddOperation(
          NNADAPTER_DEQUANTIZE, {input_operand}, {output_operand});
      break;
    }
    default:
      LOG(FATAL) << "Unsupported input precision: "
                 << static_cast<int32_t>(input_type->precision);
      break;
  }
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
