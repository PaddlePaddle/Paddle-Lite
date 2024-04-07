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

#include "driver/verisilicon_timvx/optimizer/fix_ops.h"
#include <cmath>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace verisilicon_timvx {

static void FixResizeLinearNearest(core::Model* model,
                                   core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto output_operand = output_operands[0];
  auto output_operations = GetOperandConsumers(model, output_operand);
  auto dummy_operand =
      AppendUnaryOperation(model, output_operand, NNADAPTER_RELU);
  UpdateOperationInputOperands(
      output_operations, output_operand, dummy_operand);
  UpdateModelOutputOperands(model, output_operand, dummy_operand);
}

void FixOps(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
      case NNADAPTER_RESIZE_LINEAR:
      case NNADAPTER_RESIZE_NEAREST:
        FixResizeLinearNearest(model, operation);
        break;
      default:
        break;
    }
  }
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
