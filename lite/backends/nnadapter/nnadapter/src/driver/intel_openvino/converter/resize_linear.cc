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
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "operation/resize_linear.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertResizeLinear(Converter* converter, core::Operation* operation) {
  RESIZE_LINEAR_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (input_tensor == nullptr) {
    input_tensor = converter->ConvertOperand(input_operand);
  }

  std::shared_ptr<Tensor> shape_tensor(nullptr);
  std::shared_ptr<Tensor> scale_tensor(nullptr);
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

  // Set attributes for interpolate op.
  default_opset::Interpolate::InterpolateAttrs attrs;
  attrs.mode = default_opset::Interpolate::InterpolateMode::LINEAR_ONNX;
  // attrs.nearest_mode = default_opset::Interpolate::NearestMode::FLOOR;
  if (!align_corners && align_mode == 1) {
    attrs.coordinate_transformation_mode =
        default_opset::Interpolate::CoordinateTransformMode::ASYMMETRIC;
  } else if (!align_corners && align_mode == 0) {
    attrs.coordinate_transformation_mode =
        default_opset::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
  } else if (align_corners) {
    attrs.coordinate_transformation_mode =
        default_opset::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
  }
  attrs.antialias = false;
  attrs.pads_begin = {0, 0, 0, 0};
  attrs.pads_end = {0, 0, 0, 0};

  Tensor scales;
  Tensor target_spatial_shape;
  auto shape_of_input = std::make_shared<default_opset::ShapeOf>(*input_tensor);
  if (shape_tensor) {
    // Calculate scales from ouput shape
    attrs.shape_calculation_mode =
        default_opset::Interpolate::ShapeCalcMode::SIZES;
    auto shape_begin_tensor =
        converter->AddConstantTensor(std::vector<int64_t>({0}));
    auto shape_end_tensor =
        converter->AddConstantTensor(std::vector<int64_t>({-2}));
    auto nc_shape =
        std::make_shared<default_opset::StridedSlice>(shape_of_input->output(0),
                                                      *shape_begin_tensor,
                                                      *shape_end_tensor,
                                                      std::vector<int64_t>{0},
                                                      std::vector<int64_t>{0});
    target_spatial_shape =
        std::make_shared<default_opset::Concat>(
            TensorVector{nc_shape->output(0),
                         std::make_shared<default_opset::Convert>(
                             *shape_tensor, GetElementType<int64_t>())},
            0)
            ->output(0);
    const float epsilon = 1.0e-5;
    auto converted_shape_of_input = std::make_shared<default_opset::Convert>(
        shape_of_input->output(0), GetElementType<float>());
    auto converted_sizes = std::make_shared<default_opset::Convert>(
        target_spatial_shape, GetElementType<float>());
    auto divide_op = std::make_shared<default_opset::Divide>(
        converted_sizes->output(0), converted_shape_of_input->output(0));
    auto eps_tensor = converter->AddConstantTensor<float>(epsilon);
    scales =
        std::make_shared<default_opset::Add>(divide_op->output(0), *eps_tensor)
            ->output(0);
  } else {
    // Calculate target output shape from scales.
    attrs.shape_calculation_mode =
        default_opset::Interpolate::ShapeCalcMode::SCALES;
    scales =
        std::make_shared<default_opset::Concat>(
            TensorVector{*converter->AddConstantTensor<float>({1.0f, 1.0f}),
                         *scale_tensor},
            0)
            ->output(0);
    auto converted_shape_of_input = std::make_shared<default_opset::Convert>(
        shape_of_input->output(0), scales.get_element_type());
    auto multiply_op = std::make_shared<default_opset::Multiply>(
        converted_shape_of_input->output(0), scales);
    target_spatial_shape =
        std::make_shared<default_opset::Convert>(multiply_op->output(0),
                                                 GetElementType<int64_t>())
            ->output(0);
  }
  auto interplate_op = std::make_shared<default_opset::Interpolate>(
      *input_tensor, target_spatial_shape, scales, attrs);
  MAP_OUTPUT(output_operand, interplate_op, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
