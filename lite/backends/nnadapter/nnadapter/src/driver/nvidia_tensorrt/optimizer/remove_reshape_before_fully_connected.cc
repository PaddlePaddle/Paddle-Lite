// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/nvidia_tensorrt/optimizer/remove_reshape_before_fully_connected.h"
#include <cmath>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace nvidia_tensorrt {

/*
 * IFullyConnectedLayer need 3D/4D input tensor.
 */
void RemoveReshapeBeforeFullyConnected(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    if (operation->type != NNADAPTER_RESHAPE) continue;
    auto reshape_out = operation->output_operands[0];
    if (IsModelOutputOperand(reshape_out)) continue;
    auto next_operations = GetOperandConsumers(model, reshape_out);
    if (next_operations.size() != 1) continue;
    auto next_operation = next_operations[0];
    if (next_operation->type != NNADAPTER_FULLY_CONNECTED) continue;
    next_operation->input_operands[0] = operation->input_operands[0];
    RemoveOperand(model, reshape_out);
    RemoveOperation(model, operation);
  }
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
