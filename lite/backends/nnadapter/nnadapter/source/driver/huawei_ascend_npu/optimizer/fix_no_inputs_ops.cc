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

void FixNoInputsOps(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  auto input_count = model->input_operands.size();
  auto output_count = model->output_operands.size();
  if (input_count == 0 && operations.size() == 0) {
    for (uint32_t i = 0; i < output_count; i++) {
      auto output_operand = model->output_operands[i];
      if (IsModelOutputOperand(output_operand)) {
        output_operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
        auto dummy_output_operand = AppendDummyOperation(model, output_operand);
        UpdateModelOutputOperands(model, output_operand, dummy_output_operand);
      }
    }
  }
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
