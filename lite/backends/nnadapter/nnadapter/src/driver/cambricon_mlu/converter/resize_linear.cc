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

#include "operation/resize_linear.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertResizeLinear(Converter* converter, core::Operation* operation) {
  RESIZE_LINEAR_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (input_tensor == nullptr) {
    input_tensor = converter->ConvertOperand(input_operand);
  }

  magicmind::ITensor* shape_tensor = nullptr;
  magicmind::ITensor* scale_tensor = nullptr;
  if (shape_operand != nullptr) {
    shape_tensor = converter->GetMappedTensor(shape_operand);
    if (shape_tensor == nullptr) {
      shape_tensor = converter->ConvertOperand(shape_operand);
    }
  } else {
    scale_tensor = converter->GetMappedTensor(scales_operand);
    if (scale_tensor == nullptr) {
      scale_tensor = converter->ConvertOperand(scales_operand);
    }
  }
  auto resize_linear_node = converter->network()->AddIResizeNode(
      input_tensor, shape_tensor, scale_tensor);
  NNADAPTER_CHECK(resize_linear_node) << "Failed to add resize_linear node.";

  resize_linear_node->SetMode(magicmind::IResizeMode::BILINEAR);
  bool align_center = align_mode == 0;
  // align_corners and align_center cannot be True at the same time.
  NNADAPTER_CHECK_LT(align_corners && align_center, 1);
  resize_linear_node->SetAlignCorners(align_corners);
  if (align_mode == 0) {
    resize_linear_node->SetHalfPixelCenters(true);
  } else {
    resize_linear_node->SetHalfPixelCenters(false);
  }
  magicmind::Layout input_layout =
      ConvertToMagicMindDataLayout(input_operand->type.layout);
  resize_linear_node->SetLayout(input_layout, input_layout);
  auto output_tensor = resize_linear_node->GetOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
