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

int ConvertFillConstantBatchSizeLike(Converter* converter,
                                     OpInfo* op,
                                     Scope* scope) {
  // Extract op attributes
  auto input_name = op->Input("Input").front();
  auto input_scale_name = "Input0_scale";
  std::vector<float> input_scales;
  if (op->HasInputScale(input_scale_name, true)) {
    input_scales = op->GetInputScale(input_scale_name, true);
  }
  auto dtype = op->GetAttr<int>("dtype");
  auto shape = op->GetAttr<std::vector<int>>("shape");
  float value = op->HasAttr("value") ? op->GetAttr<float>("value") : 0.0f;
  int input_dim_idx =
      op->HasAttr("input_dim_idx") ? op->GetAttr<int>("input_dim_idx") : 0;
  int output_dim_idx =
      op->HasAttr("output_dim_idx") ? op->GetAttr<int>("output_dim_idx") : 0;

  // Convert to NNAdapter operands and operation
  // Shape Operand
  auto input_operand =
      converter->AddInputOperand(scope, input_name, {}, input_scales);
  auto input_type = converter->GetOperandType(input_operand);
  shape[output_dim_idx] = input_type->dimensions.data[input_dim_idx];
  auto shape_operand = converter->AddConstantOperand(shape);
  // Value operand
  NNAdapterOperand* value_operand = nullptr;
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
    case static_cast<int32_t>(lite::core::FluidType::BOOL):
      value_operand = converter->AddConstantOperand(static_cast<bool>(value));
      break;
    default:
      LOG(FATAL) << "Not supported dtype: " << dtype;
      break;
  }
  // Out operand
  auto out_name = op->Output("Out").front();
  auto out_operand = converter->AddOutputOperand(out_name);
  // Fill operation
  converter->AddOperation(
      NNADAPTER_FILL, {shape_operand, value_operand}, {out_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
