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

#include "operation/resize_nearest.h"
#include <cmath>
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertResizeNearest(Converter* converter, core::Operation* operation) {
  RESIZE_NEAREST_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(!align_corners)
      << "Precision is not aligned when align_corners=true";

  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (input_tensor == nullptr) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto resize_layer = converter->network()->addResize(*input_tensor);
  NNADAPTER_CHECK(resize_layer);
  nvinfer1::Dims output_dims;
  output_dims.nbDims = output_operand->type.dimensions.count - 1;
  for (int32_t i = 0; i < output_dims.nbDims; i++) {
    output_dims.d[i] =
        output_operand->type.dimensions.data[i + 1] == NNADAPTER_UNKNOWN
            ? -1
            : output_operand->type.dimensions.data[i + 1];
  }
  resize_layer->setOutputDimensions(output_dims);
  resize_layer->setAlignCorners(align_corners);
  resize_layer->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
  auto output_tensor = resize_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
