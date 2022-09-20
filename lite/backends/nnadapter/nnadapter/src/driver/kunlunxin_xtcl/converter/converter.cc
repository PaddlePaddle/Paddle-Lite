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

#include "driver/kunlunxin_xtcl/converter/converter.h"
#include <utility>
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

#define REGISTER_CONVERTER(__op_type__, __func_name__) \
  extern int __func_name__(Converter* converter, core::Operation* operation);
#include "driver/kunlunxin_xtcl/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_KUNLUNXIN_XTCL_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

int Converter::Apply(core::Model* model) {
  expr_index_ = 0;
  // Create the model input exprs based on the specified name
  for (int i = 0; i < model->input_operands.size(); i++) {
    ConvertOperand(
        model->input_operands[i], {}, string_format("model_input_%d", i));
  }
  // Convert the NNAdapter operations to XTCL exprs
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
#define REGISTER_CONVERTER(__op_type__, __func_name__) \
  case NNADAPTER_##__op_type__:                        \
    __func_name__(this, operation);                    \
    break;
#include "driver/kunlunxin_xtcl/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_KUNLUNXIN_XTCL_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER
      default:
        NNADAPTER_LOG(FATAL) << "Unsupported operation("
                             << OperationTypeToString(operation->type)
                             << ") is found.";
        break;
    }
  }
  return NNADAPTER_NO_ERROR;
}

std::string Converter::GetUniqueName(const std::string& suffix) {
  return string_format("_%d_%s_", expr_index_++, suffix.c_str());
}

xtcl::xExpr Converter::GetMappedExpr(core::Operand* operand) {
  auto it = exprs_->find(operand);
  if (it != exprs_->end()) {
    return it->second.back();
  }
  return xtcl::xExpr();
}

xtcl::xExpr Converter::UpdateExprMap(core::Operand* operand, xtcl::xExpr expr) {
  auto it = exprs_->find(operand);
  if (it == exprs_->end()) {
    auto result =
        exprs_->insert(std::make_pair(operand, std::vector<xtcl::xExpr>()));
    NNADAPTER_CHECK(result.second);
    it = result.first;
  }
  it->second.push_back(expr);
  return expr;
}

xtcl::xExpr Converter::AddInputTensor(std::string name,
                                      NNAdapterOperandPrecisionCode precision,
                                      const int32_t* dimensions_data,
                                      uint32_t dimensions_count) {
  if (name.empty()) {
    name = GetUniqueName();
  }
  auto shape =
      ConvertToXTCLArray<xtcl::xIndexExpr>(dimensions_data, dimensions_count);
  auto dtype = ConvertToXTCLDataType(precision);
  return builder_->CreateTensor(name, shape, dtype);
}

xtcl::xExpr Converter::AddInputTensor(const std::string& name,
                                      NNAdapterOperandPrecisionCode precision,
                                      const std::vector<int32_t>& dimensions) {
  return AddInputTensor(name, precision, dimensions.data(), dimensions.size());
}

xtcl::xExpr Converter::AddConstantTensor(
    const void* values,
    NNAdapterOperandPrecisionCode precision,
    const std::vector<int32_t>& dimensions,
    std::string name) {
  NNADAPTER_CHECK(values)
      << "The values of constant expr should not be nullptr.";
  if (name.empty()) {
    name = GetUniqueName();
  }
  params_->emplace(std::make_pair(
      name,
      CreateXTCLNDArray(
          std::vector<int64_t>(dimensions.begin(), dimensions.end()),
          ConvertToDLDataType(precision),
          values)));
  return AddInputTensor(name, precision, dimensions);
}

xtcl::xExpr Converter::AddInt32ConstantTensor(
    const int32_t* values,
    const std::vector<int32_t>& dimensions,
    const std::string& name) {
  return AddConstantTensor(values, NNADAPTER_INT32, dimensions, name);
}

xtcl::xExpr Converter::AddInt32ConstantTensor(
    const std::vector<int32_t>& values,
    const std::vector<int32_t>& dimensions,
    const std::string& name) {
  int num_values = values.size();
  return AddInt32ConstantTensor(
      &values[0],
      dimensions.empty() ? std::vector<int32_t>({num_values}) : dimensions,
      name);
}

xtcl::xExpr Converter::AddFloat32ConstantTensor(
    const float* values,
    const std::vector<int32_t>& dimensions,
    const std::string& name) {
  return AddConstantTensor(values, NNADAPTER_FLOAT32, dimensions, name);
}

xtcl::xExpr Converter::AddFloat32ConstantTensor(
    const std::vector<float>& values,
    const std::vector<int32_t>& dimensions,
    const std::string& name) {
  int num_values = values.size();
  return AddFloat32ConstantTensor(
      &values[0],
      dimensions.empty() ? std::vector<int32_t>({num_values}) : dimensions,
      name);
}

xtcl::xExpr Converter::ConvertOperand(core::Operand* operand,
                                      std::vector<int32_t> dimensions,
                                      const std::string& name) {
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < operand->type.dimensions.count; i++) {
      dimensions.push_back(operand->type.dimensions.data[i]);
    }
  }
  xtcl::xExpr tensor;
  if (IsConstantOperand(operand)) {
    tensor = AddConstantTensor(
        operand->buffer, operand->type.precision, dimensions, name);
  } else if (IsModelInputOperand(operand)) {
    tensor = AddInputTensor(name, operand->type.precision, dimensions);
  } else {
    NNADAPTER_LOG(FATAL) << "Only constant and model input operands can be "
                            "converted to xtcl::xExpr!";
  }
  UpdateExprMap(operand, tensor);
  return tensor;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
