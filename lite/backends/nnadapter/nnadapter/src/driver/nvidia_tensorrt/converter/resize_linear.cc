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

#include "operation/resize_linear.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertResizeLinear(Converter* converter, core::Operation* operation) {
  RESIZE_LINEAR_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(!align_corners)
      << "Precision is not aligned when align_corners=true";
  NNADAPTER_CHECK(!align_mode) << "Precision is not aligned when align_mode=1";

  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (input_tensor == nullptr) {
    input_tensor = converter->ConvertOperand(input_operand);
  }

  auto resize_layer = converter->network()->addResize(*input_tensor);
  NNADAPTER_CHECK(resize_layer);

  nvinfer1::Dims output_dims;
  output_dims.nbDims = 4;
  output_dims.d[0] = input_operand->type.dimensions.data[0];
  output_dims.d[1] = input_operand->type.dimensions.data[1];
  if (scales_operand != nullptr) {
    NNADAPTER_CHECK(IsConstantOperand(scales_operand))
        << "scales operands only support constant!";
    auto scale_data = reinterpret_cast<float*>(scales_operand->buffer);
    float scale_h = scale_data[0];
    float scale_w = scale_data[1];
    if (scale_h > 0 && scale_w > 0) {
      output_dims.d[2] =
          static_cast<int>(scale_h * input_operand->type.dimensions.data[2]);
      output_dims.d[3] =
          static_cast<int>(scale_w * input_operand->type.dimensions.data[3]);
      resize_layer->setOutputDimensions(output_dims);
    } else if (shape_operand != nullptr) {
      NNADAPTER_CHECK(IsConstantOperand(shape_operand))
          << "shape operands only support constant!";
      auto shape_data = reinterpret_cast<int32_t*>(shape_operand->buffer);
      output_dims.d[2] = shape_data[0];
      output_dims.d[3] = shape_data[1];
      resize_layer->setOutputDimensions(output_dims);
    } else {
      NNADAPTER_LOG(WARNING) << "Scale data cannot be less than 0, or shape "
                                "operand is not nullprt";
      return NNADAPTER_INVALID_PARAMETER;
    }
  } else if (shape_operand != nullptr) {
    NNADAPTER_CHECK(IsConstantOperand(shape_operand))
        << "shape operands only support constant!";
    auto shape_data = reinterpret_cast<int32_t*>(shape_operand->buffer);
    output_dims.d[2] = shape_data[0];
    output_dims.d[3] = shape_data[1];
    resize_layer->setOutputDimensions(output_dims);
  } else {
    NNADAPTER_LOG(WARNING)
        << "Either shape_operand or scales_operand should be set.";
    return NNADAPTER_INVALID_PARAMETER;
  }
  resize_layer->setAlignCorners(align_corners);
  resize_layer->setResizeMode(nvinfer1::ResizeMode::kLINEAR);
  auto output_tensor = resize_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
