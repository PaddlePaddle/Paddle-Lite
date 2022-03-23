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

#include "driver/nvidia_tensorrt/converter/plugin/arg_max.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "operation/arg_min_max.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertArgMinMax(Converter* converter, core::Operation* operation) {
  ARG_MIN_MAX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  ArgMaxPluginDynamic arg_max_plugin(axis, keepdim);
  std::vector<nvinfer1::ITensor*> tensors{input_tensor};
  auto layer =
      converter->network()->addPluginV2(tensors.data(), 1, arg_max_plugin);
  NNADAPTER_CHECK(layer);
  auto output_tensor = layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
