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

#include "driver/huawei_kirin_npu/optimizer/fix_multiple_outputs_ops.h"
#include <cmath>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_kirin_npu {

void FixMultipleOutputsOps(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    auto& output_operands = operation->output_operands;
    auto output_count = output_operands.size();
    switch (operation->type) {
      case NNADAPTER_SPLIT: {
        for (uint32_t i = 0; i < output_count; i++) {
          auto output_operand = output_operands[i];
          if (IsModelOutputOperand(output_operand)) {
            auto dummy_output_operand =
                InsertDummyOperation(model, output_operand);
            UpdateOperationOutputOperands(
                operation, output_operand, dummy_output_operand);
          }
        }
      } break;
      default:
        break;
    }
  }
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
