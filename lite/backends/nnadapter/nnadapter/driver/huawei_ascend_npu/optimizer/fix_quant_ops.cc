// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/huawei_ascend_npu/optimizer/fix_quant_ops.h"
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

void FixQuantConv(hal::Model* model) {
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    if (operation->type != NNADAPTER_CONV_2D) {
      continue;
    }
    auto input_operand = operation->input_operands[0];
    if (input_operand->type.precision != NNADAPTER_QUANT_INT8_SYMM_PER_LAYER) {
      continue;
    }
    AddQuantOperation(model, input_operand);
    AddDequantOperation(model, operation->output_operands[0]);
  }
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
