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

#include "operation/split.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "driver/nvidia_tensorrt/converter/plugin/split.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertSplit(Converter* converter, core::Operation* operation) {
  SPLIT_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(IsConstantOperand(split_operand))
      << "Only support split_opeand is constant.";
  NNADAPTER_CHECK_GT(axis, 0) << "Only support axis > 0";

  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  SplitPlugin split_plugin(axis - 1, split);
  std::vector<nvinfer1::ITensor*> tensors{input_tensor};
  auto split_layer =
      converter->network()->addPluginV2(tensors.data(), 1, split_plugin);
  NNADAPTER_CHECK(split_layer);
  int split_count = split.size();
  for (uint32_t i = 0; i < split_count; i++) {
    auto output_tensor = split_layer->getOutput(i);
    converter->UpdateTensorMap(output_operands[i], output_tensor);
  }

  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
