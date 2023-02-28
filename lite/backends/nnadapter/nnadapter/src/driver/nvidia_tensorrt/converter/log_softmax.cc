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

#include "operation/log_softmax.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertLogSoftmax(Converter* converter, core::Operation* operation) {
  LOG_SOFTMAX_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_GT(axis, 0);

  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  // softmax(input, axis)
  auto softmax_layer = converter->network()->addSoftMax(*input_tensor);
  NNADAPTER_CHECK(softmax_layer);
  softmax_layer->setAxes(1 << (axis - 1));
  // log(softmax(input, axis))
  auto log_layer = converter->network()->addUnary(
      *softmax_layer->getOutput(0), nvinfer1::UnaryOperation::kLOG);
  NNADAPTER_CHECK(log_layer);
  auto output_tensor = log_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
