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

#include "driver/verisilicon_timvx/optimizer/convert_fill_like_into_eltwise_mul.h"
#include <algorithm>
#include <map>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

NNADAPTER_EXPORT void ConvertFillLikeIntoEltwiseMul(hal::Model* model) {
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    if (operation->type == NNADAPTER_FILL_LIKE) {
      auto& input_operands = operation->input_operands;
      auto& output_operands = operation->output_operands;
      NNADAPTER_CHECK_EQ(input_operands.size(), 2);
      NNADAPTER_CHECK_EQ(output_operands.size(), 1);
      auto input_operand = input_operands[0];
      auto value_operand = input_operands[1];
      RemoveOperand(model, value_operand);
      hal::Operand* dummy_operand;
      const std::vector<int32_t> dimensions({1});
      if (IsUInt8AsymmPerLayerQuantType(input_operand->type.precision)) {
        uint8_t value = 0;
        dummy_operand =
            AddQuant8ConstantOperand(model, &value, dimensions, 1.f, 128, true);
      } else {
        float value = 0.0f;
        dummy_operand =
            AddFloat32ConstantOperand(model, &value, dimensions, true);
      }
      input_operands.push_back(dummy_operand);
      operation->type = NNADAPTER_MUL;
    }
  }
}

}  // namespace nnadapter
