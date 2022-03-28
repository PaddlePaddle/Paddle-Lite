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

#include "driver/fake_device/converter/converter.h"
#include <unistd.h>
#include <algorithm>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace fake_device {

#define REGISTER_CONVERTER(                                     \
    __op_type__, __validate_func_name__, __convert_func_name__) \
  extern int __convert_func_name__(Converter* converter,        \
                                   core::Operation* operation);
#include "driver/fake_device/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_FAKE_DEVICE_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

int Converter::Apply(core::Model* model) {
  // Convert the NNAdapter operations to the fake_ddk operators
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
#define REGISTER_CONVERTER(                                     \
    __op_type__, __validate_func_name__, __convert_func_name__) \
  case NNADAPTER_##__op_type__:                                 \
    __convert_func_name__(this, operation);                     \
    break;
#include "driver/fake_device/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_FAKE_DEVICE_CONVERTER_ALL_H__
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

std::string Converter::GetTensorName(core::Operand* operand) {
  auto operand_id = OperandIdToString(operand);
  auto index = 0;
  auto it = tensors_->find(operand);
  if (it != tensors_->end()) {
    index = it->second.size();
  }
  return operand_id + string_format("_%d", index);
}

std::shared_ptr<fake_ddk::nn::Tensor> Converter::GetMappedTensor(
    core::Operand* operand) {
  auto it = tensors_->find(operand);
  if (it != tensors_->end()) {
    return it->second.back();
  }
  return nullptr;
}

std::shared_ptr<fake_ddk::nn::Tensor> Converter::UpdateTensorMap(
    core::Operand* operand, std::shared_ptr<fake_ddk::nn::Tensor> tensor) {
  auto it = tensors_->find(operand);
  if (it == tensors_->end()) {
    auto result = tensors_->insert(std::make_pair(
        operand, std::vector<std::shared_ptr<fake_ddk::nn::Tensor>>()));
    NNADAPTER_CHECK(result.second);
    it = result.first;
  }
  it->second.push_back(tensor);
  return tensor;
}

std::shared_ptr<fake_ddk::nn::Tensor> Converter::ConvertOperand(
    core::Operand* operand, std::vector<int32_t> dimensions) {
  auto tensor = CreateFakeDeviceTensor(graph_,
                                       GetTensorName(operand),
                                       &operand->type,
                                       operand->buffer,
                                       dimensions);
  NNADAPTER_CHECK(tensor);
  UpdateTensorMap(operand, tensor);
  return tensor;
}

int Converter::AddOperator(
    fake_ddk::nn::OperatorType type,
    std::vector<std::shared_ptr<fake_ddk::nn::Tensor>> input_tensors,
    std::vector<std::shared_ptr<fake_ddk::nn::Tensor>> output_tensors,
    void* attrs,
    std::string name) {
  return graph_->AddOperator(type, input_tensors, output_tensors, attrs, name);
}

}  // namespace fake_device
}  // namespace nnadapter
