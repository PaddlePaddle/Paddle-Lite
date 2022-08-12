// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/__custom__yolo_det.h"

#include "core/types.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateCustomYoloDet(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareCustomYoloDet(core::Operation* operation) {
  CUSTOM_YOLO_DET_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape of output
  auto& output_type = output_operand->type;
  output_type.dimensions.count = 2;
  output_type.dimensions.data[0] = keep_top_k;
  output_type.dimensions.data[1] = 6;
  for (uint32_t i = 0; i < input0_operand->type.dimensions.dynamic_count; i++) {
    output_type.dimensions.dynamic_data[i][0] = keep_top_k;
    output_type.dimensions.dynamic_data[i][1] = 6;
  }
  CopyOperandTypeWithPrecision(&output_type, input0_operand->type);
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteCustomYoloDet(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
