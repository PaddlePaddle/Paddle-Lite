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

#include "operation/mat_mul.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/utility.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertMatMul(Converter* converter, core::Operation* operation) {
  MAT_MUL_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // TODO(shentanyue) Support BatchMatMul later.
  NNADAPTER_CHECK_EQ(x_operand->type.dimensions.count, 2)
      << "Only support the dimension of x is 2 now.";
  NNADAPTER_CHECK_EQ(y_operand->type.dimensions.count, 2)
      << "Only support the dimension of y is 2 now.";

  // Convert to XTCL exprs
  // TODO(hong19860320) mapping NNADAPTER_MATMUL to XTCL CreateBatchMatmul or
  // CreateMatmul2D
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
