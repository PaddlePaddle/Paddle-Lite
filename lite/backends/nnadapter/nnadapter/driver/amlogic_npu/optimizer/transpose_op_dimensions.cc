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

#include "driver/amlogic_npu/optimizer/transpose_op_dimensions.h"
#include <cmath>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace amlogic_npu {

static void Transpose(hal::Model* model, hal::Operand* input_operand) {
  auto producer = GetOperandProducer(model, input_operand);
  if (producer->type == NNADAPTER_RESHAPE) {
    std::vector<int32_t> permutation;
    auto input_shape_count = input_operand->type.dimensions.count;
    for (int32_t i = input_shape_count - 1; i >= 0; --i) {
      permutation.push_back(i);
    }
    TransposeOperand(input_operand, permutation);
  }
}

void TransposeOpDimensions(hal::Model* model) {
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    auto& input_operands = operation->input_operands;
    auto& output_operands = operation->output_operands;
    switch (operation->type) {
      case NNADAPTER_FULLY_CONNECTED:
        Transpose(model, input_operands[0]);
        break;
      default:
        break;
    }
  }
}

}  // namespace amlogic_npu
}  // namespace nnadapter
