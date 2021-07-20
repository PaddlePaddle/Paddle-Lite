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

#include "driver/amlogic_npu/optimizer/unpack_op_fusion.h"
#include <cmath>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace amlogic_npu {

static void UnpackConv2D(hal::Model* model, hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 13);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto fuse_code = reinterpret_cast<int32_t*>(input_operands[10]->buffer);
  auto output_operand = output_operands[0];
  // Unpack fused activations
  if (*fuse_code != NNADAPTER_FUSED_NONE) {
    switch (*fuse_code) {
      case NNADAPTER_FUSED_RELU:
        AddUnaryOperation(model, output_operand, NNADAPTER_RELU);
        break;
      case NNADAPTER_FUSED_RELU6:
        AddUnaryOperation(model, output_operand, NNADAPTER_RELU6);
        break;
      default:
        NNADAPTER_LOG(FATAL) << "Unhandled case: fuse_code=" << *fuse_code;
        break;
    }
    *fuse_code = NNADAPTER_FUSED_NONE;
  }
}

void UnpackOpFusion(hal::Model* model) {
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
      case NNADAPTER_CONV_2D:
        UnpackConv2D(model, operation);
        break;
      default:
        break;
    }
  }
}

}  // namespace amlogic_npu
}  // namespace nnadapter
