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

#include "driver/verisilicon_timvx/converter/converter.h"
#include <unistd.h>
#include <algorithm>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace verisilicon_timvx {

#define REGISTER_CONVERTER(__op_type__, __func_name__) \
  extern int __func_name__(Converter* converter, core::Operation* operation);
#include "driver/verisilicon_timvx/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_VERISILICON_TIMVX_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

int Converter::Apply(core::Model* model) {
  // Convert the NNAdapter operations to the tim-vx operations
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
#include "driver/verisilicon_timvx/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_VERISILICON_TIMVX_CONVERTER_ALL_H__
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

std::shared_ptr<tim::vx::Tensor> Converter::GetMappedTensor(
    core::Operand* operand) {
  auto it = tensors_->find(operand);
  if (it != tensors_->end()) {
    return it->second.back();
  }
  return nullptr;
}

std::shared_ptr<tim::vx::Tensor> Converter::UpdateTensorMap(
    core::Operand* operand, std::shared_ptr<tim::vx::Tensor> tensor) {
  auto it = tensors_->find(operand);
  if (it == tensors_->end()) {
    auto result = tensors_->insert(std::make_pair(
        operand, std::vector<std::shared_ptr<tim::vx::Tensor>>()));
    NNADAPTER_CHECK(result.second);
    it = result.first;
  }
  it->second.push_back(tensor);
  return tensor;
}

std::shared_ptr<tim::vx::Tensor> Converter::AddTensor(
    const NNAdapterOperandType* type,
    void* buffer,
    std::vector<int32_t> dimensions) {
  auto tensor = CreateTimVXTensor(graph_, type, buffer, dimensions);
  NNADAPTER_CHECK(tensor);
  return tensor;
}

std::shared_ptr<tim::vx::Tensor> Converter::ConvertOperand(
    core::Operand* operand, std::vector<int32_t> dimensions) {
  auto tensor = AddTensor(&operand->type, operand->buffer, dimensions);
  UpdateTensorMap(operand, tensor);
  return tensor;
}
}  // namespace verisilicon_timvx
}  // namespace nnadapter
