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

#include "operation/unsqueeze.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertUnsqueeze(Converter* converter, core::Operation* operation) {
  UNSQUEEZE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }

  auto input_dims_data = input_operand->type.dimensions.data;
  auto input_dims_count = input_operand->type.dimensions.count;
  auto output_dims_count = input_dims_count + axes.size();
  std::vector<int32_t> output_dims(output_dims_count, 0);
  for (size_t i = 0; i < input_dims_count; i++) {
    if (input_dims_data[i] != -1) {
      output_dims[i] = input_dims_data[i];
    }
  }
  uint32_t cur_size = input_dims_count;
  for (size_t i = 0; i < axes.size(); i++) {
    int32_t axis = axes[i] < 0 ? axes[i] + cur_size + 1 : axes[i];
    for (uint32_t j = cur_size; j > axis; j--) {
      output_dims[j] = output_dims[j - 1];
    }
    output_dims[axis] = 1;
    cur_size++;
  }
  auto unsqueeze_layer = converter->network()->addShuffle(*input_tensor);
  nvinfer1::Dims reshape_dims;
  reshape_dims.nbDims = output_dims_count;
  memcpy(
      reshape_dims.d, output_dims.data(), sizeof(int32_t) * output_dims_count);
  unsqueeze_layer->setReshapeDimensions(reshape_dims);
  auto output_tensor = unsqueeze_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
