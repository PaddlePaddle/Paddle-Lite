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

#include "driver/nvidia_tensorrt/converter/plugin/hard_swish.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "operation/hard_sigmoid_swish.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertHardSwish(Converter* converter, core::Operation* operation) {
  HARD_SIGMOID_SWISH_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  HardSwishPluginDynamic hard_swish_plugin(alpha, beta);
  std::vector<nvinfer1::ITensor*> tensors{input_tensor};
  auto hard_swish_layer =
      converter->network()->addPluginV2(tensors.data(), 1, hard_swish_plugin);
  NNADAPTER_CHECK(hard_swish_layer);
  auto output_tensor = hard_swish_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
