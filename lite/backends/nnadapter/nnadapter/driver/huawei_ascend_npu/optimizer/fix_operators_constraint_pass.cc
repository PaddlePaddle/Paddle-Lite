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

#include "driver/huawei_ascend_npu/optimizer/fix_operators_constraint_pass.h"
#include <cmath>
#include <vector>
#include "core/operation/expand.h"
#include "core/operation/range.h"
#include "core/operation/reduce_mean.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int FixOperatorsConstraintPass(hal::Model* model) {
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  NNADAPTER_VLOG(5) << "Enter FixOperatorsConstraintPass";
  for (auto operation : operations) {
    switch (operation->type) {
      case NNADAPTER_EXPAND: {
        NNADAPTER_CHECK_EQ(
            ExpandShapeOperandIsConstantLimitPass(model, operation),
            NNADAPTER_NO_ERROR);
      } break;
      case NNADAPTER_RANGE: {
        NNADAPTER_CHECK_EQ(
            RangeAllOperandsIsConstantLimitPass(model, operation),
            NNADAPTER_NO_ERROR);
      } break;
      case NNADAPTER_REDUCE_MEAN: {
        NNADAPTER_CHECK_EQ(ReduceMeanAsModelOutputPass(model, operation),
                           NNADAPTER_NO_ERROR);
      } break;
      default:
        break;
    }
  }
  return NNADAPTER_NO_ERROR;
}

/*
 * ReduceMeanAsModelOutputPass:
 * In the following cases need add op trans scalar to 1D tensor with shape[1]:
 * 1. keep_dim == false && reduce_all
 * 2. output operand is model output operand.
*/
int ReduceMeanAsModelOutputPass(hal::Model* model, hal::Operation* operation) {
  NNADAPTER_VLOG(5) << "Enter ReduceMeanAsModelOutputPass";
  REDUCE_MEAN_OPERATION_EXTRACT_INPUTS_OUTPUTS
  auto reduce_all = axes_size == input_operand->type.dimension_count;
  if (!keep_dim && reduce_all && IsModelOutputOperand(output_operand)) {
    AddDummyOperation(model, output_operand);
  }
  return NNADAPTER_NO_ERROR;
}

int RangeAllOperandsIsConstantLimitPass(hal::Model* model,
                                        hal::Operation* operation) {
  NNADAPTER_VLOG(5) << "Enter RangeAllOperandsIsConstantLimitPass";
  RANGE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  for (operand : input_operands) {
    if (!IsConstantOperand(operand)) {
      NNADAPTER_LOG(ERROR) << "range input operands only support constant!";
      return NNADAPTER_INVALID_PARAMETER;
    }
  }
  return NNADAPTER_NO_ERROR;
}

int ExpandShapeOperandIsConstantLimitPass(hal::Model* model,
                                          hal::Operation* operation) {
  NNADAPTER_VLOG(5) << "Enter ExpandShapeOperandIsConstantLimitPass";
  EXPAND_OPERATION_EXTRACT_INPUTS_OUTPUTS
  if (!IsConstantOperand(shape_operand)) {
    NNADAPTER_LOG(ERROR) << "Expand shape operand only support constant!";
    return NNADAPTER_INVALID_PARAMETER;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
