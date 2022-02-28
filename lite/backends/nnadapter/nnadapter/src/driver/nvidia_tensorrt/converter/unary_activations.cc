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

#include "operation/unary_activations.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "driver/nvidia_tensorrt/plugin/relu.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertUnaryActivations(Converter* converter, core::Operation* operation) {
  UNARY_ACTIVATIONS_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  std::map<NNAdapterOperationType, nvinfer1::ActivationType>
      activation_type_map{
          {NNADAPTER_RELU, nvinfer1::ActivationType::kRELU},
          {NNADAPTER_SIGMOID, nvinfer1::ActivationType::kSIGMOID},
      };
  auto operation_type = operation->type;
  NNADAPTER_CHECK(activation_type_map.count(operation_type))
      << "Not support operation_type: "
      << OperationTypeToString(operation_type);
  auto activation_layer = converter->network()->addActivation(
      *input_tensor, activation_type_map.at(operation_type));
  // example: how to use relu plugin
  // ReluPluginDynamic relu_plugin;
  // std::vector<nvinfer1::ITensor*> tensors{input_tensor};
  // auto activation_layer =
  //     converter->network()->addPluginV2(tensors.data(), 1, relu_plugin);
  NNADAPTER_CHECK(activation_layer);
  auto output_tensor = activation_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
