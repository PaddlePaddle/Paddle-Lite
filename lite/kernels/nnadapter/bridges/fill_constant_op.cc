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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

NNAdapterOperand* ConcatOperands(Converter* converter,
                                 Scope* scope,
                                 const std::vector<std::string>& input_names,
                                 const std::string& output_name,
                                 NNAdapterOperandPrecisionCode precision) {
  NNAdapterOperand* output_operand = nullptr;
  if (converter->HasOperand(output_name)) {
    output_operand = converter->GetOperand(output_name);
  } else {
    // Concat inputs
    std::vector<NNAdapterOperand*> input_operands;
    for (size_t i = 0; i < input_names.size(); i++) {
      auto input_name = input_names[i];
      NNAdapterOperand* input_operand = nullptr;
      if (converter->HasOperand(input_name)) {
        input_operand = converter->GetOperand(input_name);
      } else {
        input_operand =
            converter->AddVariableOperand(DDim({1}), input_name, precision);
      }
      input_operands.push_back(input_operand);
    }
    // Concat axis
    auto* axis_operand = converter->AddInt32ConstantOperand(0);
    input_operands.push_back(axis_operand);
    // Concat output
    output_operand = converter->AddVariableOperand(
        DDim({static_cast<int64_t>(input_names.size())}),
        output_name,
        precision);
    std::vector<NNAdapterOperand*> output_operands = {output_operand};
    auto concat_operation = converter->AddOperation(NNADAPTER_CONCAT);
    converter->SetOperation(
        concat_operation, &input_operands, &output_operands);
  }
  return output_operand;
}

int FillConstantConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Shape operand
  NNAdapterOperand* shape_operand = nullptr;
  // Priority: ShapeTensor > ShapeTensorList > shape(attr)
  if (op_info->HasInput("ShapeTensor") &&
      !op_info->Input("ShapeTensor").empty()) {
    auto shape_name = op_info->Input("ShapeTensor").front();
    auto shape_tensor = scope->FindTensor(shape_name);
    auto shape_dims = shape_tensor->dims();
    auto shape_precision = shape_tensor->precision();
    CHECK(shape_precision == PRECISION(kInt64) ||
          shape_precision == PRECISION(kInt32))
        << "Shape's data type should be int32 or int64, but received "
        << lite_api::PrecisionToStr(shape_precision);
    shape_operand = converter->AddVariableOperand(
        shape_dims,
        shape_name,
        Precision2NNAdapterTensorPrecisionCode(shape_precision));
  } else if (op_info->HasInput("ShapeTensorList") &&
             !op_info->Input("ShapeTensorList").empty()) {
    // Use concat to generate shape_operand
    auto shape_names = op_info->Input("ShapeTensorList");
    // Warning: should use a unique name
    std::string new_shape_name = shape_names.front() + "_concat_shape";
    auto shape_precision = scope->FindTensor(shape_names.front())->precision();
    CHECK(shape_precision == PRECISION(kInt64) ||
          shape_precision == PRECISION(kInt32))
        << "Shape's data type should be int32 or int64, but received "
        << lite_api::PrecisionToStr(shape_precision);
    shape_operand =
        ConcatOperands(converter,
                       scope,
                       shape_names,
                       new_shape_name,
                       Precision2NNAdapterTensorPrecisionCode(shape_precision));
  } else if (op_info->HasAttr("shape")) {
    auto shape = op_info->GetAttr<std::vector<int64_t>>("shape");
    shape_operand = converter->AddInt64ConstantOperand(
        shape.data(), DDim({static_cast<int64_t>(shape.size())}));
  } else {
    LOG(WARNING)
        << "One of ShapeTensor, ShapeTensorList or shape(attr) must be set.";
    return FAILED;
  }

  // Value operand
  NNAdapterOperand* value_operand = nullptr;
  NNAdapterOperandPrecisionCode value_precision = NNADAPTER_TENSOR_FLOAT32;
  if (op_info->HasInput("ValueTensor") &&
      !op_info->Input("ValueTensor").empty()) {
    auto value_name = op_info->Input("ValueTensor").front();
    auto value_tensor = scope->FindTensor(value_name);
    value_precision =
        Precision2NNAdapterTensorPrecisionCode(value_tensor->precision());
    if (converter->HasOperand(value_name)) {
      value_operand = converter->GetOperand(value_name);
    } else {
      value_operand = converter->AddVariableOperand(
          value_tensor->dims(), value_name, value_precision);
    }
  } else if (op_info->HasInput("str_value") &&
             !op_info->GetAttr<std::string>("str_value").empty()) {
    LOG(WARNING) << "Not support str_value now.";
    return FAILED;
  } else if (op_info->HasAttr("value")) {
    float value = op_info->GetAttr<float>("value");
    int dtype = op_info->GetAttr<int>("dtype");
    value_precision = FluidDataType2NNAdapterTensorPrecisionCode(dtype);
    // FluidType ==> lite/core/types.h
    switch (dtype) {
      case 0: {
        bool value_bool = static_cast<bool>(value);
        value_operand =
            converter->AddBool8ConstantOperand(&value_bool, DDim({1}));
        break;
      }
      case 2: {
        int32_t value_int32 = static_cast<int32_t>(value);
        value_operand =
            converter->AddInt32ConstantOperand(&value_int32, DDim({1}));
        break;
      }
      case 3: {
        int64_t value_int64 = static_cast<int64_t>(value);
        value_operand =
            converter->AddInt64ConstantOperand(&value_int64, DDim({1}));
        break;
      }
      case 5: {
        value_operand = converter->AddFloat32ConstantOperand(&value, DDim({1}));
        break;
      }
      default:
        LOG(WARNING) << "Not support dtype: " << dtype;
        return FAILED;
    }
  } else {
    LOG(WARNING)
        << "One of ValueTensor, str_value(attr) or value(attr) must be set.";
    return FAILED;
  }

  // Output operand
  auto output_name = op_info->Output("Out").front();
  NNAdapterOperand* output_operand = converter->AddVariableOperand(
      scope->FindTensor(output_name)->dims(), output_name, value_precision);

  // Fill operation
  std::vector<NNAdapterOperand*> input_operands = {shape_operand,
                                                   value_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  auto fill_operation = converter->AddOperation(NNADAPTER_FILL);
  converter->SetOperation(fill_operation, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    fill_constant,
    kNNAdapter,
    paddle::lite::subgraph::nnadapter::FillConstantConverter);
