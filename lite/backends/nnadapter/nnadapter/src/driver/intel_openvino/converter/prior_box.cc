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
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "operation/prior_box.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertPriorBox(Converter* converter, core::Operation* operation) {
  PRIOR_BOX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto image_tensor = converter->GetMappedTensor(image_operand);
  if (!image_tensor) {
    image_tensor = converter->ConvertOperand(image_operand);
  }

  auto make_slice = [converter](
      const std::shared_ptr<Operator>& op,
      int64_t start,
      int64_t end) -> std::shared_ptr<default_opset::StridedSlice> {
    return std::make_shared<default_opset::StridedSlice>(
        op->output(0),
        *converter->AddConstantTensor(std::vector<int64_t>({start})),
        *converter->AddConstantTensor(std::vector<int64_t>({end})),
        std::vector<int64_t>{0},   // begin mask
        std::vector<int64_t>{0});  // end mask
  };

  auto input_shape = std::make_shared<default_opset::ShapeOf>(*input_tensor);
  auto Image_shape = std::make_shared<default_opset::ShapeOf>(*image_tensor);
  auto output_shape_slice = make_slice(input_shape, 2, 4);
  auto image_shape_slice = make_slice(Image_shape, 2, 4);

  float step_w = *reinterpret_cast<float*>(step_w_operand->buffer);
  float step_h = *reinterpret_cast<float*>(step_h_operand->buffer);
  NNADAPTER_CHECK_EQ(step_w, step_h)
      << "Only supports one step for ov prior_box op";
  // Set prior_box attrs.
  default_opset::PriorBox::Attributes attrs;
  attrs.min_size = min_sizes;
  attrs.max_size = max_sizes;
  attrs.aspect_ratio = aspect_ratios;
  attrs.step = step_w;
  attrs.flip = *reinterpret_cast<bool*>(flip_operand->buffer);
  attrs.clip = *reinterpret_cast<bool*>(clip_operand->buffer);
  attrs.min_max_aspect_ratios_order =
      *reinterpret_cast<bool*>(min_max_aspect_ratios_order_operand->buffer);
  attrs.offset = *reinterpret_cast<float*>(offset_operand->buffer);
  attrs.variance = variances;

  auto prior_box_op = std::make_shared<default_opset::PriorBox>(
      output_shape_slice->output(0), image_shape_slice->output(0), attrs);
  auto out_shape = std::make_shared<default_opset::Concat>(
      TensorVector{output_shape_slice->output(0),
                   *converter->AddConstantTensor<int64_t>({-1, 4})},
      0);
  auto split_axis_tensor = converter->AddConstantTensor<int64_t>(0);
  auto prior_box_split_op = std::make_shared<default_opset::Split>(
      prior_box_op->output(0), *split_axis_tensor, 2);
  auto boxes_reshape_op = std::make_shared<default_opset::Reshape>(
      prior_box_split_op->output(0), out_shape->output(0), true);
  auto variances_reshape_op = std::make_shared<default_opset::Reshape>(
      prior_box_split_op->output(1), out_shape->output(0), true);
  MAP_OUTPUT(boxes_operand, boxes_reshape_op, 0);
  MAP_OUTPUT(Variances_operand, variances_reshape_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
