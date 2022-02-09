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
  // Input operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);

  // Value operand
  NNAdapterOperand* value_operand = nullptr;
  float value = op->GetAttr<float>("value");
  int dtype = op->GetAttr<int>("dtype");
  auto input_precision = converter->GetOperandType(input_operand)->precision;
  if (dtype == -1) {
    switch (input_precision) {
      case NNADAPTER_FLOAT32:
        dtype = static_cast<int32_t>(lite::core::FluidType::FP32);
        break;
      case NNADAPTER_INT32:
        dtype = static_cast<int32_t>(lite::core::FluidType::INT32);
        break;
      case NNADAPTER_INT64:
        dtype = static_cast<int32_t>(lite::core::FluidType::INT64);
        break;
      case NNADAPTER_INT8:
        dtype = static_cast<int32_t>(lite::core::FluidType::INT8);
        break;
      default:
        LOG(FATAL) << "Not supported x dtype: "
                   << static_cast<int>(NNADAPTER_FLOAT32);
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
    case static_cast<int32_t>(lite::core::FluidType::INT8):
      value_operand =
          converter->AddConstantOperand(static_cast<int8_t>(value), x_scales);
      break;
    default:
      LOG(FATAL) << "Not supported dtype: " << dtype;
      break;
  }

  // Output operand
  auto out_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
  }
  NNAdapterOperand* output_operand =
      converter->AddOutputOperand(out_name, out_scales);

  // Fill operation
  converter->AddOperation(
      NNADAPTER_FILL_LIKE, {input_operand, value_operand}, {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
