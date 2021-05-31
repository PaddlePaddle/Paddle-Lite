// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>
#include "nnadapter_types.h"  // NOLINT

namespace nnadapter {

// Utilities for strings
std::string string_format(const std::string fmt_str, ...);

// Utilities for convert NNAdapter type to string
std::string ResultCodeToString(NNAdapterResultCode type);
std::string OperandPrecisionCodeToString(NNAdapterOperandPrecisionCode type);
std::string OperandLayoutCodeToString(NNAdapterOperandLayoutCode type);
std::string OperandLifetimeCodeToString(NNAdapterOperandLifetimeCode type);
std::string OperationTypeToString(NNAdapterOperationType type);
std::string FuseCodeToString(NNAdapterFuseCode type);
std::string DeviceCodeToString(NNAdapterDeviceCode type);
std::string DimensionsToString(const int32_t* dimensions,
                               uint32_t dimension_count);
int OperandPrecisionLength(NNAdapterOperandPrecisionCode type);
std::string OperandPrecisionName(NNAdapterOperandPrecisionCode type);
bool IsPerLayerQuantization(NNAdapterOperandPrecisionCode type);
bool IsPerChannelQuantization(NNAdapterOperandPrecisionCode type);
bool IsAsymmetricQuantization(NNAdapterOperandPrecisionCode type);
bool IsSymmetricQuantization(NNAdapterOperandPrecisionCode type);
bool IsAsymmPerLayerQuantization(NNAdapterOperandPrecisionCode type);
bool IsSymmPerLayerQuantization(NNAdapterOperandPrecisionCode type);
bool IsSymmPerChannelQuantization(NNAdapterOperandPrecisionCode type);
bool IsUInt8AsymmPerLayerQuantization(NNAdapterOperandPrecisionCode type);
bool IsInt8SymmPerLayerQuantization(NNAdapterOperandPrecisionCode type);
bool IsInt8SymmPerChannelQuantization(NNAdapterOperandPrecisionCode type);
bool IsUInt32AsymmPerLayerQuantization(NNAdapterOperandPrecisionCode type);
bool IsInt32SymmPerLayerQuantization(NNAdapterOperandPrecisionCode type);
bool IsInt32SymmPerChannelQuantization(NNAdapterOperandPrecisionCode type);

// Utilities for NNAdapter dimensions
int64_t ProductionOfDimensions(const int32_t* input_dimensions,
                               uint32_t input_dimension_count);
int64_t ProductionOfDimensions(const std::vector<int32_t>& input_dimensions);
void TransposeDimensions(int32_t* input_dimensions,
                         const std::vector<int32_t>& permutation,
                         int32_t* output_dimensions_ptr = nullptr);
void ReshapeDimensions(int32_t* input_dimensions,
                       uint32_t* input_dimension_count,
                       const std::vector<int32_t>& dimensions,
                       int32_t* output_dimensions_ptr = nullptr,
                       uint32_t* output_dimension_count_ptr = nullptr);

}  // namespace nnadapter
