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

#include "operation/elementwise.h"
#include "driver/google_xnnpack/converter/converter.h"
#include "driver/google_xnnpack/converter/validator.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace google_xnnpack {

bool ValidateElementwise(Validator* validator,
                         const core::Operation* operation) {
  return true;
}

int ConvertElementwise(Converter* converter, core::Operation* operation) {
  ELEMENTWISE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XNNPACK tensor value ids and nodes
  auto input0_tensor_value_id =
      converter->GetMappedTensorValueId(input0_operand);
  if (input0_tensor_value_id == XNN_INVALID_VALUE_ID) {
    input0_tensor_value_id = converter->ConvertOperand(input0_operand);
  }
  auto input1_tensor_value_id =
      converter->GetMappedTensorValueId(input1_operand);
  if (input1_tensor_value_id == XNN_INVALID_VALUE_ID) {
    input1_tensor_value_id = converter->ConvertOperand(input1_operand);
  }
  auto output_tensor_value_id = converter->ConvertOperand(output_operand);
  float output_min, output_max;
  ConvertFuseCodeToXNNClippingRange(fuse_code, &output_min, &output_max);
  if (operation->type == NNADAPTER_ADD) {
    ADD_OPERATOR(xnn_define_add2,
                 output_min,
                 output_max,
                 input0_tensor_value_id,
                 input1_tensor_value_id,
                 output_tensor_value_id,
                 0);
  } else if (operation->type == NNADAPTER_SUB) {
    ADD_OPERATOR(xnn_define_subtract,
                 output_min,
                 output_max,
                 input0_tensor_value_id,
                 input1_tensor_value_id,
                 output_tensor_value_id,
                 0);
  } else if (operation->type == NNADAPTER_MUL) {
    ADD_OPERATOR(xnn_define_multiply2,
                 output_min,
                 output_max,
                 input0_tensor_value_id,
                 input1_tensor_value_id,
                 output_tensor_value_id,
                 0);
  } else if (operation->type == NNADAPTER_DIV) {
    ADD_OPERATOR(xnn_define_divide,
                 output_min,
                 output_max,
                 input0_tensor_value_id,
                 input1_tensor_value_id,
                 output_tensor_value_id,
                 0);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported element-wise operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace google_xnnpack
}  // namespace nnadapter
