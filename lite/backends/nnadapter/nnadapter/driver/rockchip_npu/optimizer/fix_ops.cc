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

#include "driver/rockchip_npu/optimizer/fix_ops.h"
#include <cmath>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace rockchip_npu {

static void FixRELUDepthwiseConv2D(hal::Model* model,
                                   hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 1);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto output_operand = output_operands[0];
  // Because rknpu_ddk has a bug in the case of RELU+DepthwiseConv2D, Check if
  // the consumers has a depthwise conv2d operation and insert a dummy ADD
  // operation
  auto operations = GetOperandConsumers(model, output_operand);
  for (auto& operation : operations) {
    if (operation->type != NNADAPTER_CONV_2D) continue;
    NNADAPTER_CHECK_GT(output_operand->type.dimension_count, 1);
    auto group =
        *reinterpret_cast<int32_t*>(operation->input_operands[9]->buffer);
    auto is_depthwise_mode =
        (group != 1 && output_operand->type.dimensions[1] == group);
    if (is_depthwise_mode) {
      auto dummy_add_operand = AddOperand(model);
      memcpy(&dummy_add_operand->type,
             &output_operand->type,
             sizeof(NNAdapterOperandType));
      InsertOperand(model, output_operand, dummy_add_operand, true);
      int8_t dummy_addend_value = 0;
      auto dummy_addend_operand = AddQuant8ConstantOperand(
          model, &dummy_addend_value, std::vector<int32_t>({1}), 0.0f);
      auto dummy_fuse_code_operand = AddInt32ConstantOperand(model, 0);
      auto dummy_add_operation = AddOperation(model);
      dummy_add_operation->type = NNADAPTER_ADD;
      dummy_add_operation->input_operands = {
          output_operand, dummy_addend_operand, dummy_fuse_code_operand};
      dummy_add_operation->output_operands = {dummy_add_operand};
      break;
    }
  }
}

void FixOps(hal::Model* model) {
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
      case NNADAPTER_RELU:
        FixRELUDepthwiseConv2D(model, operation);
        break;
      default:
        break;
    }
  }
}

}  // namespace rockchip_npu
}  // namespace nnadapter
