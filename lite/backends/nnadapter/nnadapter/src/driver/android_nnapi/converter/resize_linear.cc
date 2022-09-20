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

#include "operation/resize_linear.h"
#include <memory>
#include "driver/android_nnapi/converter/converter.h"
#include "driver/android_nnapi/converter/validator.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace android_nnapi {

bool ValidateResizeLinear(Validator* validator,
                          const core::Operation* operation) {
  return true;
}

int ConvertResizeLinear(Converter* converter, core::Operation* operation) {
  RESIZE_LINEAR_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to NNAPI operands and operations
  auto input_index = converter->GetMappedIndex(input_operand);
  if (input_index == INVALID_INDEX) {
    input_index = converter->ConvertOperand(input_operand);
  }
  auto output_index = converter->ConvertOperand(output_operand);
  // NHWC
  auto output_width = output_operand->type.dimensions.data[2];
  NNADAPTER_CHECK_NE(output_width, NNADAPTER_UNKNOWN)
      << "The output width should not be dynamic!";
  auto output_height = output_operand->type.dimensions.data[1];
  NNADAPTER_CHECK_NE(output_height, NNADAPTER_UNKNOWN)
      << "The output height should not be dynamic!";
  auto output_width_index = converter->AddInt32ConstantOperand(output_width);
  auto output_height_index = converter->AddInt32ConstantOperand(output_height);
  std::vector<uint32_t> input_indexes(
      {input_index, output_width_index, output_height_index});
  if (nnapi()->android_sdk_version < ANEURALNETWORKS_FEATURE_LEVEL_4) {
    NNADAPTER_CHECK(!align_corners)
        << "The align_corners=true are only supported after Android "
        << ANEURALNETWORKS_FEATURE_LEVEL_4 << " but the runtime's is "
        << nnapi()->android_sdk_version;
  } else if (align_corners) {
    auto is_nchw_index = converter->AddBool8ConstantOperand(false);
    auto align_corners_index =
        converter->AddBool8ConstantOperand(align_corners);
    input_indexes.push_back(is_nchw_index);
    input_indexes.push_back(align_corners_index);
    // TODO(hong19860320) Check align_mode
  }
  NNADAPTER_CHECK_EQ(
      converter->AddOperation(
          ANEURALNETWORKS_RESIZE_BILINEAR, input_indexes, {output_index}),
      ANEURALNETWORKS_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace android_nnapi
}  // namespace nnadapter
