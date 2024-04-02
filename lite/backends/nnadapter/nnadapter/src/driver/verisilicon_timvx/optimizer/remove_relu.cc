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

#include "driver/verisilicon_timvx/optimizer/remove_relu.h"
#include <algorithm>
#include <map>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace verisilicon_timvx {

// Convert input(scale,zero_point=128)->relu->output to
// input(scale,zero_point=0)
NNADAPTER_EXPORT void RemoveRelu(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    if (operation->type != NNADAPTER_RELU) continue;
    auto relu_input_operand = operation->input_operands[0];
    auto relu_output_operand = operation->output_operands[0];
    if (IsModelInputOperand(relu_input_operand)) continue;
    auto relu_input_consumers = GetOperandConsumers(model, relu_input_operand);
    if (relu_input_consumers.size() != 1) continue;
    if (!IsUInt8AsymmPerLayerQuantType(relu_input_operand->type.precision) ||
        !IsUInt8AsymmPerLayerQuantType(relu_output_operand->type.precision))
      continue;
    relu_input_operand->type.asymm_per_layer_params.scale =
        relu_output_operand->type.asymm_per_layer_params.scale;
    relu_input_operand->type.asymm_per_layer_params.zero_point = 0;
    auto relu_output_consumers =
        GetOperandConsumers(model, relu_output_operand);
    UpdateOperationInputOperands(
        relu_output_consumers, relu_output_operand, relu_input_operand);
    if (IsModelOutputOperand(relu_output_operand)) {
      UpdateModelOutputOperands(model, relu_output_operand, relu_input_operand);
    }
    RemoveOperand(model, relu_output_operand);
    RemoveOperation(model, operation);
  }
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
