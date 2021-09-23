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

int ConvertFillConstant(Converter* converter, OpInfo* op, Scope* scope) {
  // Shape operand
  NNAdapterOperand* shape_operand = nullptr;
  // Priority: ShapeTensor > ShapeTensorList > shape(attr)
  if (op->HasInput("ShapeTensor") && !op->Input("ShapeTensor").empty()) {
    auto shape_name = op->Input("ShapeTensor").front();
    auto shape_precision = scope->FindTensor(shape_name)->precision();
    CHECK(shape_precision == PRECISION(kInt64) ||
          shape_precision == PRECISION(kInt32))
        << "Shape's data type should be int32 or int64, but received "
        << lite_api::PrecisionToStr(shape_precision);
    shape_operand = converter->AddInputOperand(scope, shape_name);
  } else if (op->HasInput("ShapeTensorList") &&
             !op->Input("ShapeTensorList").empty()) {
    // TODO(zhupengyang): Use concat to generate shape_operand later
    LOG(WARNING) << "Not support ShapeTensorList now.";
    return PARAMETER_ERROR;
  } else if (op->HasAttr("shape")) {
    std::vector<int64_t> shape = op->GetAttr<std::vector<int64_t>>("shape");
    shape_operand = converter->AddConstantOperand(shape);
  } else {
    LOG(WARNING)
        << "One of ShapeTensor, ShapeTensorList or shape(attr) must be set.";
    return PARAMETER_ERROR;
  }

  // Value operand
  NNAdapterOperand* value_operand = nullptr;
  if (HasInput(op, scope, "ValueTensor")) {
    auto value_name = op->Input("ValueTensor").front();
    value_operand = converter->AddInputOperand(scope, value_name);
  } else if (op->HasInput("str_value") &&
             !op->GetAttr<std::string>("str_value").empty()) {
    // TODO(zhupengyang): support str_value later
    LOG(WARNING) << "Not support str_value now.";
    return PARAMETER_ERROR;
  } else if (op->HasAttr("value")) {
    float value = op->GetAttr<float>("value");
    int dtype = op->GetAttr<int>("dtype");
    // FluidType ==> lite/core/types.h
    switch (dtype) {
      case static_cast<int>(lite::core::FluidType::BOOL):
        value_operand = converter->AddConstantOperand(static_cast<bool>(value));
        break;
      case static_cast<int>(lite::core::FluidType::INT32):
        value_operand =
            converter->AddConstantOperand(static_cast<int32_t>(value));
        break;
      case static_cast<int>(lite::core::FluidType::INT64):
        value_operand =
            converter->AddConstantOperand(static_cast<int64_t>(value));
        break;
      case static_cast<int>(lite::core::FluidType::FP32):
        value_operand =
            converter->AddConstantOperand(static_cast<float>(value));
        break;
      default:
        LOG(WARNING) << "Not support dtype: " << dtype;
        return PARAMETER_ERROR;
    }
  } else {
    LOG(WARNING)
        << "One of ValueTensor, str_value(attr) or value(attr) must be set.";
    return PARAMETER_ERROR;
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
