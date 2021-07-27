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

#include "driver/huawei_ascend_npu/optimizer/fix_no_inputs_ops.h"
#include <cmath>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

void FixNoInputsOps(hal::Model* model) {
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    // auto& output_operands = operation->output_operands;
    NNADAPTER_VLOG(5) << "[fix] type:" << operation->type;
    // auto output_count = output_operands.size();
    // if (operation->type == nullptr) {
    //   auto output_operand = output_operands[0];
    //   if (IsConstantOperand(output_operand)) {
    //     AddDummyOperation(model, output_operand);
    //   }
    // }
  }
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
