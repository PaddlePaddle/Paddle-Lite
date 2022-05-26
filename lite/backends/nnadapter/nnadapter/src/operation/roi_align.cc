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

#include "operation/roi_align.h"
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateRoiAlign(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareRoiAlign(core::Operation* operation) {
  ROI_ALIGN_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto input_type = input_operand->type;
  auto rois_type = rois_operand->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeExceptQuantParams(&output_type, input_type);

  auto infer_output_shape = [&](int32_t* input_dimensions_data,
                                int32_t* rois_dimensions_data,
                                int32_t* output_dimensions_data) {
    output_dimensions_data[0] = rois_dimensions_data[0];
    output_dimensions_data[1] = input_dimensions_data[1];
    output_dimensions_data[2] = output_height;
    output_dimensions_data[3] = output_width;
  };

  infer_output_shape(input_type.dimensions.data,
                     rois_type.dimensions.data,
                     output_type.dimensions.data);
  for (uint32_t i = 0; i < input_type.dimensions.dynamic_count; i++) {
    infer_output_shape(input_type.dimensions.dynamic_data[i],
                       rois_type.dimensions.dynamic_data[i],
                       output_type.dimensions.dynamic_data[i]);
  }

  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteRoiAlign(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
