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

#include "driver/huawei_ascend_npu/optimizer/fix_reduce_ops_scalar_output.h"
#include <cmath>
#include <vector>
#include "operation/reduce.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

/*
 * ReduceMeanAsModelOutputFixScalarOutput:
 * In the following cases need add op trans scalar to 1D tensor with shape[1]:
 * 1. keep_dim == false && reduce_all
 * 2. output operand is model output operand.
*/
static void ReduceOpsAddDummyOperation(core::Model* model,
                                       core::Operation* operation) {
  REDUCE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  auto reduce_all =
      axes_size == static_cast<int>(input_operand->type.dimensions.count);
  if (!keep_dim && reduce_all && IsModelOutputOperand(output_operand)) {
    auto dummy_output_operand = InsertDummyOperation(model, output_operand);
    UpdateOperationOutputOperands(
        operation, output_operand, dummy_output_operand);
  }
}

void FixReduceOpsScalarOutput(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    switch (operation->type) {
      case NNADAPTER_REDUCE_MEAN:
        ReduceOpsAddDummyOperation(model, operation);
        break;
      default:
        break;
    }
  }
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
