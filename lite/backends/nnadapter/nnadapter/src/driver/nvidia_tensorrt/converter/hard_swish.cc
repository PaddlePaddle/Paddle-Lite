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
#include "utility/modeling.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertHardSwish(Converter* converter, core::Operation* operation) {
  HARD_SIGMOID_SWISH_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto hard_sigmoid_layer = converter->network()->addActivation(
      *input_tensor, nvinfer1::ActivationType::kHARD_SIGMOID);
  NNADAPTER_CHECK(hard_sigmoid_layer);
  hard_sigmoid_layer->setAlpha(alpha);
  hard_sigmoid_layer->setBeta(beta);
  auto hard_swish_layer = converter->network()->addElementWise(
      *input_tensor,
      *hard_sigmoid_layer->getOutput(0),
      nvinfer1::ElementWiseOperation::kPROD);
  NNADAPTER_CHECK(hard_swish_layer);
  // Here is an example to use plugin
  // std::vector<nvinfer1::ITensor*> tensors{input_tensor};
  // nvinfer1::IPluginV2Layer* hard_swish_layer = nullptr;
  // if (IsOperandWithDynamicShape(input_operand)) {
  //   HardSwishPluginDynamic hard_swish_plugin(alpha, beta);
  //   hard_swish_layer =
  //       converter->network()->addPluginV2(tensors.data(), 1,
  //       hard_swish_plugin);
  // } else {
  //   HardSwishPlugin hard_swish_plugin(alpha, beta);
  //   hard_swish_layer =
  //       converter->network()->addPluginV2(tensors.data(), 1,
  //       hard_swish_plugin);
  // }
  converter->UpdateTensorMap(output_operand, hard_swish_layer->getOutput(0));
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
