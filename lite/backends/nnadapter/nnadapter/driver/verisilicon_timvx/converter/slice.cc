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

#include "core/operation/slice.h"
#include "driver/verisilicon_timvx/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace verisilicon_timvx {

int ConvertSlice(Converter* converter, hal::Operation* operation) {
  SLICE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to tim-vx tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);
  auto output_dim = output_operand->type.dimensions.count;
  std::vector<int32_t> axis;
  for (uint32_t i = 0; i < axes_count; i++) {
    axis.push_back(output_dim - 1 - axes[i]);
  }
  std::vector<int32_t> start;
  for (int i = 0; i < output_dim; i++) {
    bool match_axis = false;
    for (uint32_t j = 0; j < axes_count; j++) {
      if (i == axis[j]) {
        match_axis = true;
        break;
      }
    }
    if (match_axis)
      start.push_back(starts[i]);
    else
      start.push_back(0);
  }
  std::vector<int32_t> length;
  for (int i = 0; i < output_dim; i++) {
    bool match_axis = false;
    for (uint32_t j = 0; j < axes_count; j++) {
      if (i == axis[j]) {
        match_axis = true;
        break;
      }
    }
    if (match_axis)
      start.push_back(ends[i] - starts[i]);
    else
      start.push_back(output_operand->type.dimensions.data[output_dim - 1 - i]);
  }
  auto slice_op = converter->graph()->CreateOperation<tim::vx::ops::Slice>(
      output_dim, start, length);
  slice_op->BindInputs({input_tensor});
  slice_op->BindOutputs({output_tensor});
  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
