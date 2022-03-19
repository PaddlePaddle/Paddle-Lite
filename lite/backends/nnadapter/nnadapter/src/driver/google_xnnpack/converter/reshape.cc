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

#include "operation/reshape.h"
#include "driver/google_xnnpack/converter/converter.h"
#include "driver/google_xnnpack/converter/validator.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace google_xnnpack {

bool ValidateReshape(Validator* validator, const core::Operation* operation) {
  return true;
}

int ConvertReshape(Converter* converter, core::Operation* operation) {
  RESHAPE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XNNPACK tensor value ids and nodes
  auto input_tensor_value_id = converter->GetMappedTensorValueId(input_operand);
  if (input_tensor_value_id == XNN_INVALID_VALUE_ID) {
    input_tensor_value_id = converter->ConvertOperand(input_operand);
  }
  auto output_tensor_value_id = converter->ConvertOperand(output_operand);
  NNADAPTER_CHECK(IsConstantOperand(shape_operand))
      << "Only supports the constant shape!";
  auto shape_count = shape_operand->length / sizeof(int32_t);
  auto shape_data = reinterpret_cast<int32_t*>(shape_operand->buffer);
  std::vector<size_t> shape(shape_data, shape_data + shape_count);
  ADD_OPERATOR(xnn_define_static_reshape,
               shape.size(),
               shape.data(),
               input_tensor_value_id,
               output_tensor_value_id,
               0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace google_xnnpack
}  // namespace nnadapter
