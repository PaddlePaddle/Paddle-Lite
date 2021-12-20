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

#include "driver/kunlunxin_xtcl/converter/converter.h"
#include <utility>
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

#define REGISTER_CONVERTER(__op_type__, __func_name__) \
  extern int __func_name__(Converter* converter, hal::Operation* operation);
#include "driver/kunlunxin_xtcl/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_KUNLUNXIN_XTCL_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

int Converter::Apply(hal::Model* model) {
  expr_index_ = 0;
  // Convert the NNAdapter operations to XTCL exprs
  std::vector<hal::Operation*> operations =
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

xtcl::xExpr Converter::GetMappedExpr(hal::Operand* operand) {
  auto it = exps_->find(operand);
  if (it != exprs_->end()) {
    return it->second.back();
  }
  return nullptr;
}

xtcl::xExpr Converter::UpdateExprMap(hal::Operand* operand, xtcl::xExpr expr) {
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

xtcl::xExpr Converter::AddConstantExpr(const void* values,
                                       NNAdapterOperandPrecisionCode precision,
                                       const std::vector<int32_t>& dimensions) {
  NNADAPTER_CHECK(values)
      << "The values of constant expr should not be nullptr.";
  auto num_values = ProductionOfDimensions(dimensions);
  // auto shape = ConvertToArrayOf<xtcl::xIndexExpr>(dimensions);
  // auto dtype = ConvertToDataType(precision);
  // auto constant_expr = builder_->CreateTensor(name, shape, dtype);
  // params_.emplace(std::make_pair(name, ConvertToxNDArray(tensor, shape,
  // layout)));
  // Add anonymous constant operator
  return nullptr;
}

xtcl::xExpr Converter::AddInt32ConstantExpr(
    const int32_t* values, const std::vector<int32_t>& dimensions) {
  return AddConstantOperator(values, NNADAPTER_INT32, dimensions);
}

xtcl::xExpr Converter::AddInt32ConstantExpr(
    const std::vector<int32_t>& values,
    const std::vector<int32_t>& dimensions) {
  int num_values = values.size();
  return AddInt32ConstantExpr(
      &values[0],
      dimensions.empty() ? std::vector<int32_t>({num_values}) : dimensions);
}

xtcl::xExpr Converter::AddFloat32ConstantExpr(
    const float* values, const std::vector<int32_t>& dimensions) {
  return AddConstantExpr(values, NNADAPTER_FLOAT32, dimensions);
}

xtcl::xExpr Converter::AddFloat32ConstantExpr(
    const std::vector<float>& values, const std::vector<int32_t>& dimensions) {
  int num_values = values.size();
  return AddFloat32ConstantExpr(
      &values[0],
      dimensions.empty() ? std::vector<int32_t>({num_values}) : dimensions);
}

xtcl::xExpr Converter::ConvertOperand(hal::Operand* operand,
                                      std::vector<int32_t> dimensions) {
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < operand->type.dimensions.count; i++) {
      dimensions.push_back(operand->type.dimensions.data[i]);
    }
  }
  NNADAPTER_LOG(FATAL) << "Only constant and model input operands can be "
                          "converted to xtcl::xExpr!";
  return nullptr;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
