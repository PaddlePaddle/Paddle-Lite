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

#include "driver/rockchip_npu/optimizer/unpack_op_fusion.h"
#include <cmath>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace rockchip_npu {

static void UnpackActivations(core::Model* model,
                              core::Operation* operation,
                              core::Operand* output_operand,
                              core::Operand* fuse_code_operand) {
  auto fuse_code = reinterpret_cast<int32_t*>(fuse_code_operand->buffer);
  // Prevent the op fusion of conv2d and relu6 to solve the precision problem
  if (*fuse_code == NNADAPTER_FUSED_RELU6) {
    *fuse_code = NNADAPTER_FUSED_NONE;
    // Insert a relu6 operation
    auto act_output_operand =
        InsertUnaryOperation(model, output_operand, NNADAPTER_RELU6);
    UpdateOperationOutputOperands(
        operation, output_operand, act_output_operand);
  }
}

void UnpackOpFusion(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    auto& input_operands = operation->input_operands;
    auto& output_operands = operation->output_operands;
    switch (operation->type) {
      case NNADAPTER_CONV_2D:
        UnpackActivations(
            model, operation, output_operands[0], input_operands[8]);
        break;
      default:
        break;
    }
  }
}

}  // namespace rockchip_npu
}  // namespace nnadapter
