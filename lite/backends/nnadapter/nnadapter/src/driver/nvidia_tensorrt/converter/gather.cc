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

#include "operation/gather.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertGather(Converter* converter, core::Operation* operation) {
  GATHER_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_GT(axis, 0) << "Only support axis > 0";

  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto indices_tensor = converter->GetMappedTensor(indices_operand);
  if (!indices_tensor) {
    indices_tensor = converter->ConvertOperand(indices_operand);
  }
  // Reshape layer
  // auto reshape_layer = converter->network()->addShuffle(*indices_tensor);
  // NNADAPTER_CHECK(reshape_layer);
  // nvinfer1::Dims indices_shape{};
  // indices_shape.nbDims = 1;
  // indices_shape.d[0] = -1;
  // reshape_layer->setReshapeDimensions(indices_shape);
  // Gather layer
  auto gather_layer =
      converter->network()->addGather(*input_tensor, *indices_tensor, axis - 1);
  NNADAPTER_CHECK(gather_layer);
  // gather_layer->setNbElementWiseDims(0);
  auto output_tensor = gather_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
