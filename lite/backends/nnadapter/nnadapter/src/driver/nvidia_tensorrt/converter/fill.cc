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

#include "driver/nvidia_tensorrt/converter/plugin/fill.h"
#include <iostream>
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "operation/fill.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertFill(Converter* converter, core::Operation* operation) {
  FILL_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(IsConstantOperand(shape_operand));
  NNADAPTER_CHECK(IsConstantOperand(value_operand));

  // Convert to trt tensors and node
  if (value_operand->type.precision == NNADAPTER_FLOAT32) {
    auto value = reinterpret_cast<float*>(value_operand->buffer);
    size_t size = value_operand->length / sizeof(float);
    auto weight = converter->AddWeights(value, size);
    auto dims = ConvertToNVDims(output_operand->type.dimensions, false);
    auto constant_layer = converter->network()->addConstant(dims, weight);
    NNADAPTER_CHECK(constant_layer);
    converter->UpdateTensorMap(output_operand, constant_layer->getOutput(0));
  } else {
    NNADAPTER_LOG(FATAL) << "Only support float32 value";
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
