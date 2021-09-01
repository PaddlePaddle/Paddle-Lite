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

int ConvertFillAnyLike(Converter* converter, OpInfo* op, Scope* scope) {
  // Use "shape" + "fill" to implement "fill_any_like"
  // Shape operand
  auto x_name = op->Input("X").front();
  auto out_name = op->Output("Out").front();
  auto shape_operand = converter->AddShapeOperation(x_name, out_name);

  // Value operand
  NNAdapterOperand* value_operand = nullptr;
  float value = op->GetAttr<float>("value");
  int dtype = op->GetAttr<int>("dtype");
  auto* input_operand = converter->GetMappedOperand(x_name);
  auto input_precision = converter->GetOperandType(input_operand)->precision;
  if (dtype == -1) {
    switch (input_precision) {
      case NNADAPTER_TENSOR_FLOAT32:
        dtype = static_cast<int32_t>(lite::core::FluidType::FP32);
        break;
      case NNADAPTER_TENSOR_INT32:
        dtype = static_cast<int32_t>(lite::core::FluidType::INT32);
        break;
      case NNADAPTER_TENSOR_INT64:
        dtype = static_cast<int32_t>(lite::core::FluidType::INT64);
        break;
      default:
        LOG(FATAL) << "Not supported x dtype: "
                   << static_cast<int>(NNADAPTER_TENSOR_FLOAT32);
        break;
    }
  }
  switch (dtype) {
    case static_cast<int32_t>(lite::core::FluidType::FP32):
      value_operand = converter->AddConstantOperand(value);
      break;
    case static_cast<int32_t>(lite::core::FluidType::INT32):
      value_operand =
          converter->AddConstantOperand(static_cast<int32_t>(value));
      break;
    case static_cast<int32_t>(lite::core::FluidType::INT64):
      value_operand =
          converter->AddConstantOperand(static_cast<int64_t>(value));
      break;
    default:
      LOG(FATAL) << "Not supported dtype: " << dtype;
      break;
  }

  // Output operand
  NNAdapterOperand* output_operand = converter->AddOutputOperand(out_name);

  // Fill operation
  std::vector<NNAdapterOperand*> input_operands = {shape_operand,
                                                   value_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  converter->AddOperation(NNADAPTER_FILL, &input_operands, &output_operands);
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
