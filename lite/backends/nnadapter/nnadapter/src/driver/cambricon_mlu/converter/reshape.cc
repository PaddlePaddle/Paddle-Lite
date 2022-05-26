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

#include "operation/reshape.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertReshape(Converter* converter, core::Operation* operation) {
  RESHAPE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  if (IsTemporaryShapeOperand(shape_operand)) {
    auto& temporary_shape = *(GetTemporaryShape(shape_operand));
    auto shape_count = temporary_shape.count;
    auto shape_data = temporary_shape.data;
    for (uint32_t i = 0; i < shape_count; i++) {
      if (shape_data[i] == 0) {
        if (input_operand->type.dimensions.data[i] == NNADAPTER_UNKNOWN) {
          shape_data[i] = -1;
        } else {
          shape_data[i] = input_operand->type.dimensions.data[i];
        }
      }
    }
  } else if (IsConstantOperand(shape_operand)) {
    auto shape_count = shape_operand->length / sizeof(int32_t);
    auto shape_data = reinterpret_cast<int32_t*>(shape_operand->buffer);
    for (uint32_t i = 0; i < shape_count; i++) {
      if (shape_data[i] == 0) {
        if (input_operand->type.dimensions.data[i] == NNADAPTER_UNKNOWN) {
          shape_data[i] = -1;
        } else {
          shape_data[i] = input_operand->type.dimensions.data[i];
        }
      }
    }
  }

  auto shape_tensor = converter->ConvertOperand(shape_operand);
  auto reshape_node =
      converter->network()->AddIReshapeNode(input_tensor, shape_tensor);
  NNADAPTER_CHECK(reshape_node) << "Failed to add reshape node.";
  auto output_tensor = reshape_node->GetOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
