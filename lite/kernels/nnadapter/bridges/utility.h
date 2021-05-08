// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

bool HasInput(const OpInfo* op_info,
              const Scope* scope,
              const std::string& arg_name);

bool HasOutput(const OpInfo* op_info,
               const Scope* scope,
               const std::string& arg_name);

bool IsPerChannelScales(const std::vector<float>& scales);

bool IsPrecisionCompatible(const NNAdapterOperandType* target,
                           const PrecisionType reference);

bool IsLayoutCompatible(const NNAdapterOperandType* target,
                        const DataLayoutType& reference);

bool IsDimensionCompatible(const NNAdapterOperandType* target,
                           const DDim& reference);

// Paddle to NNAdapter
std::vector<int32_t> ConvertDimensions(const DDim& input_dimensions);
// NNAdapter to Paddle
DDim ConvertDimensions(int32_t* input_dimensions,
                       uint32_t input_dimension_count);

// NNAdapter to Paddle
PrecisionType ConvertPrecision(NNAdapterOperandPrecisionCode input_precision);

template <typename T>
void Quantize(const float* input_data,
              size_t input_size,
              const std::vector<float>& input_scale,
              T* output_data) {
  bool per_layer = input_scale.size() == 1;
  CHECK(per_layer || input_size == input_scale.size())
      << "Only input_scale.size() == 1 and input_scale.size() == input_size is "
         "supported.";
  int quant_bits = sizeof(T) * 8;
  auto dtype_max = static_cast<int>((1 << (quant_bits - 1)) - 1);
  auto dtype_min = static_cast<int>(0 - dtype_max);
  for (size_t i = 0; i < input_size; i++) {
    int scale_index = per_layer ? 0 : i;
    output_data[i] = std::min(
        std::max(static_cast<T>(input_data[i] / input_scale[scale_index]),
                 dtype_min),
        dtype_max);
  }
}

template <typename T>
void Dequantize(const T* input_data,
                size_t input_size,
                const std::vector<float>& input_scale,
                float* output_data) {
  bool per_layer = input_scale.size() == 1;
  CHECK(per_layer || input_size == input_scale.size())
      << "Only input_scale.size() == 1 and input_scale.size() == input_size is "
         "supported.";
  int quant_bits = sizeof(T) * 8;
  auto dtype_max = static_cast<int>((1 << (quant_bits - 1)) - 1);
  auto dtype_min = static_cast<int>(0 - dtype_max);
  for (size_t i = 0; i < input_size; i++) {
    int scale_index = per_layer ? 0 : i;
    output_data[i] = std::min(std::max(input_data[i], dtype_min), dtype_max) *
                     input_scale[scale_index];
  }
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
