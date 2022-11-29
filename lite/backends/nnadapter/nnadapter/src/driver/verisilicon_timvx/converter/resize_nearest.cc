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

#include "operation/resize_nearest.h"
#include "driver/verisilicon_timvx/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace verisilicon_timvx {

int ConvertResizeNearest(Converter* converter, core::Operation* operation) {
  RESIZE_NEAREST_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to tim-vx tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  if (shape_operand != nullptr) {
    auto shape_tensor = converter->GetMappedTensor(shape_operand);
    if (shape_tensor == nullptr) {
      shape_tensor = converter->ConvertOperand(shape_operand);
    }
  }
  auto output_tensor = converter->ConvertOperand(output_operand);

  float factor = 0.0f;
  if (scales_operand) {
    NNADAPTER_CHECK(IsConstantOperand(scales_operand));
    factor = reinterpret_cast<float*>(scales_operand->buffer)[0] ==
                     reinterpret_cast<float*>(scales_operand->buffer)[1]
                 ? reinterpret_cast<float*>(scales_operand->buffer)[0]
                 : 0;
  } else {
    NNADAPTER_CHECK(
        shape_operand && IsConstantOperand(shape_operand) &&
        (input_operand->type.dimensions.data[2] != NNADAPTER_UNKNOWN));
    factor =
        static_cast<float>(output_operand->type.dimensions.data[2] /
                           input_operand->type.dimensions.data[2]) ==
                static_cast<float>(output_operand->type.dimensions.data[3] /
                                   input_operand->type.dimensions.data[3])
            ? static_cast<float>(output_operand->type.dimensions.data[2] /
                                 input_operand->type.dimensions.data[2])
            : 0;
  }
  bool half_pixel_centers =
      (static_cast<float>(output_operand->type.dimensions.data[2] /
                          input_operand->type.dimensions.data[2]) > 4) ||
      (static_cast<float>(output_operand->type.dimensions.data[3] /
                          input_operand->type.dimensions.data[3]) > 4);
  auto resize_op = converter->graph()->CreateOperation<tim::vx::ops::Resize>(
      tim::vx::ResizeType::NEAREST_NEIGHBOR,
      factor,
      align_corners,
      half_pixel_centers,
      output_operand->type.dimensions.data[2],
      output_operand->type.dimensions.data[3]);
  resize_op->BindInputs({input_tensor});
  resize_op->BindOutputs({output_tensor});
  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
