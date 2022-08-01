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
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "operation/fill.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace nvidia_tensorrt {

template <typename T>
nvinfer1::Weights GenerateWeight(core::Operand* value_operand,
                                 const NNAdapterOperandDimensionType& dims,
                                 Converter* converter) {
  auto value_data = reinterpret_cast<T*>(value_operand->buffer);
  size_t value_size = value_operand->length / sizeof(T);
  int64_t size = ProductionOfDimensions(dims.data, dims.count);
  if (value_size != static_cast<size_t>(size)) {
    NNADAPTER_CHECK_EQ(value_size, 1U);
    std::vector<T> value(size, *value_data);
    auto weight = converter->AddWeights(value);
    return weight;
  } else {
    auto weight = converter->AddWeights(value_data, value_size);
    return weight;
  }
}

int ConvertFill(Converter* converter, core::Operation* operation) {
  FILL_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(IsConstantOperand(shape_operand));
  NNADAPTER_CHECK(IsConstantOperand(value_operand));

  // Convert to trt tensors and node
  auto precision = value_operand->type.precision;
  auto out_dims = output_operand->type.dimensions;
  nvinfer1::Weights weight;
  if (precision == NNADAPTER_FLOAT32) {
    weight = GenerateWeight<float>(value_operand, out_dims, converter);
  } else {
    NNADAPTER_LOG(FATAL) << "Not support precision: "
                         << OperandPrecisionCodeToString(precision);
  }
  auto dims = ConvertToNVDims(output_operand->type.dimensions, false);
  auto constant_layer = converter->network()->addConstant(dims, weight);
  NNADAPTER_CHECK(constant_layer);
  converter->UpdateTensorMap(output_operand, constant_layer->getOutput(0));
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
