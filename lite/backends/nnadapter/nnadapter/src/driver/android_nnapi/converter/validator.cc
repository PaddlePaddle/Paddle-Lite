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

#include "driver/android_nnapi/converter/validator.h"
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace android_nnapi {

#define REGISTER_CONVERTER(                                     \
    __op_type__, __validate_func_name__, __convert_func_name__) \
  extern bool __validate_func_name__(Validator* validator,      \
                                     const core::Operation* operation);
#include "driver/android_nnapi/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_ANDROID_NNAPI_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

int Validator::Apply(const core::Model* model, bool* supported_operations) {
  std::unordered_map<const core::Operation*, size_t> operation_to_index;
  size_t operation_index = 0;
  for (auto& operation : model->operations) {
    operation_to_index[&operation] = operation_index++;
  }
  auto operations = SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Validating " << OperationTypeToString(operation->type)
                      << " ...";
    bool flag = false;
    switch (operation->type) {
#define REGISTER_CONVERTER(                                     \
    __op_type__, __validate_func_name__, __convert_func_name__) \
  case NNADAPTER_##__op_type__:                                 \
    flag = __validate_func_name__(this, operation);             \
    break;
#include "driver/android_nnapi/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_ANDROID_NNAPI_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER
      default:
        break;
    }
    if (!flag) {
      NNADAPTER_LOG(WARNING) << "Unsupported operation("
                             << OperationTypeToString(operation->type)
                             << ") is found.";
    }
    supported_operations[operation_to_index[operation]] = flag;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace android_nnapi
}  // namespace nnadapter
