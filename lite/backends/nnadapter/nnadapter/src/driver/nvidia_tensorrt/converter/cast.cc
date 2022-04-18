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

#include "driver/nvidia_tensorrt/converter/plugin/cast.h"
#include <iostream>
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "operation/cast.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertCast(Converter* converter, core::Operation* operation) {
  CAST_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto input_precision = input_operand->type.precision;
  std::vector<nvinfer1::ITensor*> tensors{input_tensor};
  nvinfer1::IPluginV2Layer* cast_layer = nullptr;
  if (IsOperandWithDynamicShape(input_operand)) {
    CastPluginDynamic cast_plugin_dynamic(ConvertToNVDataType(input_precision),
                                          ConvertToNVDataType(dtype));
    cast_layer = converter->network()->addPluginV2(
        tensors.data(), 1, cast_plugin_dynamic);
  } else {
    CastPlugin cast_plugin(ConvertToNVDataType(input_precision),
                           ConvertToNVDataType(dtype));
    cast_layer =
        converter->network()->addPluginV2(tensors.data(), 1, cast_plugin);
  }
  NNADAPTER_CHECK(cast_layer);
  auto output_tensor = cast_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
