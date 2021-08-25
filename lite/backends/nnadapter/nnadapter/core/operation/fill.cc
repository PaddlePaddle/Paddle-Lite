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

#include "core/operation/fill.h"
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareFill(hal::Operation* operation) {
  FILL_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Infer the shape and type of output operands
  NNADAPTER_CHECK_EQ(shape_operand->type.lifetime, NNADAPTER_TEMPORARY_SHAPE);
  //   if(shape_operand->length>0){

  //   }
  //   int32_t shape_size = input_operand->type.dimension_count;
  //   output_operand->type.dimensions[0] = shape_size;
  //   output_operand->type.dynamic_dimension_count =
  //       input_operand->type.dynamic_dimension_count;
  //   for (uint32_t i = 0; i < input_operand->type.dynamic_dimension_count;
  //   i++) {
  //     output_operand->type.dynamic_dimensions[i][0] = shape_size;
  //   }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
