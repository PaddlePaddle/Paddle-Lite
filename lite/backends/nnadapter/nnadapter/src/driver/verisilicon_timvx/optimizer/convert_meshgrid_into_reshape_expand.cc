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

#include "driver/verisilicon_timvx/optimizer/convert_meshgrid_into_reshape_expand.h"
#include <algorithm>
#include <map>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

NNADAPTER_EXPORT void ConvertMeshgridIntoReshapeExpand(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    if (operation->type == NNADAPTER_MESHGRID) {
      auto& input_operands = operation->input_operands;
      auto& output_operands = operation->output_operands;
      auto input_count = input_operands.size();
      auto output_count = output_operands.size();
      NNADAPTER_CHECK_EQ(input_count, output_count);

      std::vector<core::Operand*> origin_input_operands(input_operands);

      for (int i = 0; i < output_count; i++) {
        auto input_operand = origin_input_operands[i];
        auto output_operand = output_operands[i];
        auto reshape_operand = AddOperand(model);
        std::vector<int32_t> output_shape(
            output_operand->type.dimensions.data,
            output_operand->type.dimensions.data +
                output_operand->type.dimensions.count);
        std::vector<int32_t> operand_shape(output_count, 1);
        operand_shape[i] = output_shape[i];
        auto shape_operand = AddInt32ConstantOperand(model, operand_shape);
        auto output_shape_operand =
            AddInt32ConstantOperand(model, output_shape);

        CopyOperandType(&reshape_operand->type, output_operand->type);
        for (int i = 0; i < output_count; i++) {
          reshape_operand->type.dimensions.data[i] = operand_shape[i];
        }
        reshape_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;

        if (i == 0) {
          operation->type = NNADAPTER_RESHAPE;
          operation->input_operands = {input_operand, shape_operand};
          operation->output_operands = {reshape_operand};
        } else {
          auto reshape_operation = AddOperation(model);
          reshape_operation->type = NNADAPTER_RESHAPE;
          reshape_operation->input_operands = {input_operand, shape_operand};
          reshape_operation->output_operands = {reshape_operand};
        }

        auto broadcast_operation = AddOperation(model);
        broadcast_operation->type = NNADAPTER_EXPAND;
        broadcast_operation->input_operands = {reshape_operand,
                                               output_shape_operand};
        broadcast_operation->output_operands = {output_operand};
      }
    }
  }
}

}  // namespace nnadapter
