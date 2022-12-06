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

#include "optimizer/remove_identity_transpose.h"
#include <algorithm>
#include <map>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

NNADAPTER_EXPORT void RemoveIdentityTranspose(core::Model* model) {
  NNADAPTER_VLOG(5) << "Apply RemoveIdentityTranspose";
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    if (operation->type != NNADAPTER_TRANSPOSE) continue;
    auto transpose_input_operand = operation->input_operands[0];
    auto transpose_output_operand = operation->output_operands[0];
    if (IsModelInputOperand(transpose_input_operand) &&
        IsModelOutputOperand(transpose_output_operand))
      continue;
    auto transpose_perm_operand = operation->input_operands[1];
    if (!IsConstantOperand(transpose_perm_operand)) continue;
    auto transpose_perm_count =
        transpose_perm_operand->length / sizeof(int32_t);
    auto transpose_perm_data =
        reinterpret_cast<int32_t*>(transpose_perm_operand->buffer);
    std::vector<int32_t> transpose_perm(
        transpose_perm_data, transpose_perm_data + transpose_perm_count);
    if (!IsIdentityPermutation(transpose_perm)) continue;
    auto transpose_output_consumers =
        GetOperandConsumers(model, transpose_output_operand);
    UpdateOperationInputOperands(transpose_output_consumers,
                                 transpose_output_operand,
                                 transpose_input_operand);
    if (IsModelOutputOperand(transpose_output_operand)) {
      UpdateModelOutputOperands(
          model, transpose_output_operand, transpose_input_operand);
    }
    RemoveOperand(model, transpose_output_operand);
    RemoveOperation(model, operation);
  }
}

}  // namespace nnadapter
