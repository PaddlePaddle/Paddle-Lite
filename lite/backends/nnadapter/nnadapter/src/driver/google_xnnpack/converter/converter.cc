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

#include "driver/google_xnnpack/converter/converter.h"
#include <algorithm>
#include <utility>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace google_xnnpack {

#define REGISTER_CONVERTER(                                     \
    __op_type__, __validate_func_name__, __convert_func_name__) \
  extern int __convert_func_name__(Converter* converter,        \
                                   core::Operation* operation);
#include "driver/google_xnnpack/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_GOOGLE_XNNPACK_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

int Converter::Apply(core::Model* model) {
  operand_index_ = 0;
  // Convert the NNAdapter operations to the NNAPI operations
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
#include "driver/google_xnnpack/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_GOOGLE_XNNPACK_CONVERTER_ALL_H__
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

uint32_t Converter::GetMappedTensorValueId(core::Operand* operand) {
  auto it = tensor_value_ids_->find(operand);
  if (it != tensor_value_ids_->end()) {
    return it->second.back();
  }
  return INVALID_TENSOR_VALUE_ID;
}

uint32_t Converter::UpdateTensorValueIdMap(core::Operand* operand,
                                           uint32_t index) {
  auto it = tensor_value_ids_->find(operand);
  if (it == tensor_value_ids_->end()) {
    auto result = tensor_value_ids_->insert(
        std::make_pair(operand, std::vector<uint32_t>()));
    NNADAPTER_CHECK(result.second);
    it = result.first;
  }
  it->second.push_back(index);
  return index;
}

uint32_t Converter::ConvertOperand(core::Operand* operand,
                                   std::vector<int32_t> dimensions) {
  auto& type = operand->type;
  auto buffer = operand->buffer;
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < type.dimensions.count; i++) {
      dimensions.push_back(type.dimensions.data[i]);
    }
  }
  uint32_t tensor_value_id = 0;
  NNADAPTER_CHECK_NE(tensor_value_id, INVALID_TENSOR_VALUE_ID);
  UpdateTensorValueIdMap(operand, tensor_value_id);
  return tensor_value_id;
}

}  // namespace google_xnnpack
}  // namespace nnadapter
