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

#include "driver/intel_openvino/converter/converter.h"
#include <unistd.h>
#include <algorithm>
#include <utility>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace intel_openvino {

#define REGISTER_CONVERTER(__op_type__, __func_name__) \
  extern int __func_name__(Converter* converter, core::Operation* operation);
#include "driver/intel_openvino/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_INTEL_OPENVINO_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

int Converter::Apply(core::Model* model) {
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
#include "driver/intel_openvino/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_INTEL_OPENVINO_CONVERTER_ALL_H__
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

std::shared_ptr<Tensor> Converter::GetMappedTensor(core::Operand* operand) {
  auto it = tensor_map_->find(operand);
  if (it != tensor_map_->end()) {
    return it->second.back();
  }
  return nullptr;
}

std::shared_ptr<Tensor> Converter::UpdateTensorMap(
    core::Operand* operand, std::shared_ptr<Tensor> tensor) {
  auto it = tensor_map_->find(operand);
  if (it == tensor_map_->end()) {
    auto result = tensor_map_->insert(
        std::make_pair(operand, std::vector<std::shared_ptr<Tensor>>()));
    NNADAPTER_CHECK(result.second);
    it = result.first;
  }
  tensor->set_names({OperandIdToString(operand)});
  it->second.push_back(tensor);
  return tensor;
}

std::shared_ptr<Tensor> Converter::ConvertOperand(
    core::Operand* operand, std::vector<int32_t> dimensions) {
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < operand->type.dimensions.count; i++) {
      dimensions.push_back(operand->type.dimensions.data[i]);
    }
  }
  if (IsConstantOperand(operand)) {
    auto constant_op = std::make_shared<default_opset::Constant>(
        ConvertToOVElementType(operand->type.precision),
        ConvertToOVShape(dimensions),
        operand->buffer);
    auto output_tensor = std::make_shared<Tensor>(constant_op->output(0));
    UpdateTensorMap(operand, output_tensor);
    return output_tensor;
  } else if (IsModelInputOperand(operand)) {
    auto parameter_node = std::make_shared<default_opset::Parameter>(
        ConvertToOVElementType(operand->type.precision),
        ConvertToOVShape(dimensions));
    parameter_node_map_->insert({operand, parameter_node});
    auto output_tensor = std::make_shared<Tensor>(parameter_node->output(0));
    UpdateTensorMap(operand, output_tensor);
    return output_tensor;
  }
  NNADAPTER_LOG(FATAL) << "Only constant and model input operands can be "
                          "converted to OpenVINO Tensor!";
  return nullptr;
}

// Filter's layer is [cout, cin , k_w, k_h],
// Divide cout with groups,
// final filter is [groups, cout / groups, cin, k_w, k_h].
std::shared_ptr<Operator> Converter::GetGroupConvFilterShape(
    std::shared_ptr<Tensor> filter, const int groups) {
  auto axis_index = std::make_shared<default_opset::Constant>(
      GetElementType<int64_t>(), Shape({}), std::vector<int64_t>({0}));
  auto filter_cout_index = std::make_shared<default_opset::Constant>(
      GetElementType<int64_t>(), Shape({1}), std::vector<int64_t>({0}));
  auto filter_ihw_index = std::make_shared<default_opset::Constant>(
      GetElementType<int64_t>(), Shape({3}), std::vector<int64_t>({1, 2, 3}));
  auto shape_op = std::make_shared<default_opset::ShapeOf>(*filter);
  auto filter_cout_tensor =
      std::make_shared<default_opset::Gather>(shape_op->output(0),
                                              filter_cout_index->output(0),
                                              axis_index->output(0))
          ->output(0);
  auto filter_ihw_tensor =
      std::make_shared<default_opset::Gather>(shape_op->output(0),
                                              filter_ihw_index->output(0),
                                              axis_index->output(0))
          ->output(0);
  auto groups_tensor =
      std::make_shared<default_opset::Constant>(
          GetElementType<int64_t>(), Shape({1}), std::vector<int64_t>({groups}))
          ->output(0);
  auto group_num_tensor =
      std::make_shared<default_opset::Divide>(filter_cout_tensor, groups_tensor)
          ->output(0);
  auto target_filter_shape = std::make_shared<default_opset::Concat>(
      TensorVector{groups_tensor, group_num_tensor, filter_ihw_tensor}, 0);
  return std::make_shared<default_opset::Reshape>(
      *filter, target_filter_shape->output(0), false);
}

}  // namespace intel_openvino
}  // namespace nnadapter
