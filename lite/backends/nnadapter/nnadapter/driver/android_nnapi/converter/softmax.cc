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

#include "core/operation/softmax.h"
#include "driver/android_nnapi/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace android_nnapi {

int ConvertSoftmax(Converter* converter, hal::Operation* operation) {
  SOFTMAX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to NNAPI operands and operations
  auto input_index = converter->GetMappedIndex(input_operand);
  if (input_index == INVALID_INDEX) {
    input_index = converter->ConvertOperand(input_operand);
  }
  auto beta_index = converter->AddFloat32ConstantOperand(1.0f);
  auto output_index = converter->ConvertOperand(output_operand);
  std::vector<uint32_t> input_indexes({input_index, beta_index});
  if (nnapi()->android_sdk_version < ANDROID_NNAPI_MIN_API_LEVEL_FOR_NNAPI_12) {
    auto input_dimensions_count = input_operand->type.dimensions.count;
    NNADAPTER_CHECK(
        (input_dimensions_count == 2 && (axis == 1 || axis == -1)) ||
        (input_dimensions_count == 4 && axis == 1))
        << "The 2D(axis = -1 or axis = 1) and 4D(axis = 1) input tensors are "
           "only supported before Android "
        << ANDROID_NNAPI_MIN_API_LEVEL_FOR_NNAPI_12 << " but the runtime's is "
        << nnapi()->android_sdk_version
        << " and rank = " << input_dimensions_count << ", axis = " << axis;
  } else {
    auto axis_index = converter->AddInt32ConstantOperand(axis);
    input_indexes.push_back(axis_index);
  }
  NNADAPTER_CHECK_EQ(
      converter->AddOperation(
          ANEURALNETWORKS_SOFTMAX, input_indexes, {output_index}),
      ANEURALNETWORKS_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace android_nnapi
}  // namespace nnadapter
