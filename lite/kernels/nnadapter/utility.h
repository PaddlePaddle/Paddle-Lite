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
#include "lite/backends/nnadapter/nnadapter_wrapper.h"
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

// Check whether the input or output tensors corresponding to the argument name
// exists
bool HasInput(const OpInfo* op_info,
              const Scope* scope,
              const std::string& arg_name);
bool HasOutput(const OpInfo* op_info,
               const Scope* scope,
               const std::string& arg_name);

// Check whether the two quantization parameters are the same
bool IsValidSymmQuantParams(const std::vector<float>& quant_scales,
                            uint32_t quant_channel_dim = 0);
bool IsValidSymmPerLayerQuantParams(const std::vector<float>& quant_scales);
// Check whether all of scale values are the same to determine whether it is a
// per-channel quantization method
bool IsValidSymmPerChannelQuantParams(const std::vector<float>& quant_scales,
                                      uint32_t quant_channel_dim = 0,
                                      float threshold = 1e-5f);
bool IsSameSymmQuantParams(const std::vector<float>& target_quant_scales,
                           const std::vector<float>& ref_quant_scale,
                           int32_t target_quant_channel_dim = 0,
                           int32_t ref_quant_channel_dim = 0,
                           float threshold = 1e-5f);

// Caculate the production of the given dimensions
int64_t ProductionOfDimensions(const int32_t* input_dimensions,
                               uint32_t input_dimension_count);
int64_t ProductionOfDimensions(const std::vector<int32_t>& input_dimensions);

// Quantize the fp32 data to the symmetric per-layer/per-channel quantization
// data
template <typename T>
void SymmQuantizeData(const float* input_data,
                      size_t input_size,
                      const std::vector<float>& input_scale,
                      T* output_data) {
  bool per_layer = input_scale.size() == 1;
  CHECK(per_layer || input_size == input_scale.size())
      << "Only input_scale.size() == 1 and input_scale.size() == input_size is "
         "supported.";
  int quant_bits = sizeof(T) * 8;
  auto dtype_max = static_cast<T>((1 << (quant_bits - 1)) - 1);
  auto dtype_min = static_cast<T>(0 - dtype_max);
  for (size_t i = 0; i < input_size; i++) {
    int scale_index = per_layer ? 0 : i;
    output_data[i] = std::min(
        std::max(static_cast<T>(input_data[i] / input_scale[scale_index]),
                 dtype_min),
        dtype_max);
  }
}

// Dequantize the symmetric per-layer/per-channel quantization data to the fp32
// data
template <typename T>
void SymmDequantizeData(const T* input_data,
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

// Check NNAdapter quantization type
bool IsNNSymmQuantType(NNAdapterOperandPrecisionCode precision_code);
bool IsNNSymmPerLayerQuantType(NNAdapterOperandPrecisionCode precision_code);
bool IsNNSymmPerChannelQuantType(NNAdapterOperandPrecisionCode precision_code);
bool IsNNInt8SymmQuantType(NNAdapterOperandPrecisionCode precision_code);
bool IsNNInt8SymmPerLayerQuantType(
    NNAdapterOperandPrecisionCode precision_code);
bool IsNNInt8SymmPerChannelQuantType(
    NNAdapterOperandPrecisionCode precision_code);
bool IsNNInt16SymmQuantType(NNAdapterOperandPrecisionCode precision_code);
bool IsNNInt16SymmPerLayerQuantType(
    NNAdapterOperandPrecisionCode precision_code);
bool IsNNInt16SymmPerChannelQuantType(
    NNAdapterOperandPrecisionCode precision_code);
bool IsNNInt32SymmQuantType(NNAdapterOperandPrecisionCode precision_code);
bool IsNNInt32SymmPerLayerQuantType(
    NNAdapterOperandPrecisionCode precision_code);
bool IsNNInt32SymmPerChannelQuantType(
    NNAdapterOperandPrecisionCode precision_code);

bool IsNNSymmQuantType(const NNAdapterOperandType& operand_type);
bool IsNNSymmPerLayerQuantType(const NNAdapterOperandType& operand_type);
bool IsNNSymmPerChannelQuantType(const NNAdapterOperandType& operand_type);
bool IsNNInt8SymmQuantType(const NNAdapterOperandType& operand_type);
bool IsNNInt8SymmPerLayerQuantType(const NNAdapterOperandType& operand_type);
bool IsNNInt8SymmPerChannelQuantType(const NNAdapterOperandType& operand_type);
bool IsNNInt16SymmQuantType(const NNAdapterOperandType& operand_type);
bool IsNNInt16SymmPerLayerQuantType(const NNAdapterOperandType& operand_type);
bool IsNNInt16SymmPerChannelQuantType(const NNAdapterOperandType& operand_type);
bool IsNNInt32SymmQuantType(const NNAdapterOperandType& operand_type);
bool IsNNInt32SymmPerLayerQuantType(const NNAdapterOperandType& operand_type);
bool IsNNInt32SymmPerChannelQuantType(const NNAdapterOperandType& operand_type);
bool GetNNSymmQuantParams(const NNAdapterOperandType& operand_type,
                          std::vector<float>* quant_scales,
                          uint32_t* quant_channel_dim = nullptr);
bool SetNNSymmQuantParams(NNAdapterOperandType* operand_type,
                          const std::vector<float>& quant_scales,
                          uint32_t quant_channel_dim = 0);
// Get the data length according to the NNAdapter precision code
int64_t GetNNOperandPrecisionDataLength(
    NNAdapterOperandPrecisionCode precision_code);
int64_t GetNNOperandPrecisionDataLength(
    const NNAdapterOperandType& operand_type);
// Get the data length according to the NNAdapter precision code
int64_t GetNNOperandTypeBufferLength(const NNAdapterOperandType& operand_type);

// Convert a C/C++ POD types to a NNAdapter precision code
template <typename T>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode(
    const float* quant_scales = nullptr,
    uint32_t quant_scale_count = 0,
    uint32_t quant_channel_dim = 0) {
  LOG(FATAL) << "Unable to convert a POD type(" << typeid(T).name()
             << ") to a NNAdapter precision code";
  return static_cast<NNAdapterOperandPrecisionCode>(NNADAPTER_UNKNOWN);
}
template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<bool>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim);
template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<int8_t>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim);
template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<uint8_t>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim);
template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<int16_t>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim);
template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<uint16_t>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim);
template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<int32_t>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim);
template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<uint32_t>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim);
template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<int64_t>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim);
template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<uint64_t>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim);
template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<float>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim);
template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<double>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim);
// Convert a PaddleFluid data type to a NNAdapter precision code
NNAdapterOperandPrecisionCode ConvertFluidDataTypeToNNPrecisionCode(
    int fluid_dtype,
    const float* quant_scales = nullptr,
    uint32_t quant_scale_count = 0,
    uint32_t quant_channel_dim = 0);
// Convert a PaddleLite precision type to a NNAdapter precision code
NNAdapterOperandPrecisionCode ConvertPrecisionTypeToNNPrecisionCode(
    PrecisionType precision_type,
    const float* quant_scales = nullptr,
    uint32_t quant_scale_count = 0,
    uint32_t quant_channel_dim = 0);
// Convert a PaddleLite unary activation type to a NNAdapter operation type
NNAdapterOperationType ConvertUnaryActTypeToNNOperationType(
    const std::string& unary_act_op_type);
// Convert the attribute 'padding_algorithm' in Conv2d/DepthwiseConv2d to
// NNAdapterAutoPadCode
NNAdapterAutoPadCode ConvertPaddingAlgorithmToNNAutoPadCode(
    const std::string& padding_algorithm);
// Convert the attribute 'mode' in Pad2d/Pad3d to NNAdapterPadModeCode
NNAdapterPadModeCode ConvertPadModeToNNPadModeCode(std::string mode);
// Convert the attribute 'interpolate mode' in GridSample to
// NNAdapterInterpolateModeCode
NNAdapterInterpolateModeCode ConvertInterpolateModeToNNInterpolateModeCode(
    std::string mode);

// Convert a C/C++ POD types to a PaddleLite precision type
template <typename T>
PrecisionType ConvertPODTypeToPrecisionType() {
  LOG(FATAL) << "Unable to convert a POD type(" << typeid(T).name()
             << ") to a PaddleLite precision type";
  return PRECISION(kUnk);
}
template <>
PrecisionType ConvertPODTypeToPrecisionType<bool>();
template <>
PrecisionType ConvertPODTypeToPrecisionType<int8_t>();
template <>
PrecisionType ConvertPODTypeToPrecisionType<uint8_t>();
template <>
PrecisionType ConvertPODTypeToPrecisionType<int16_t>();
template <>
PrecisionType ConvertPODTypeToPrecisionType<uint16_t>();
template <>
PrecisionType ConvertPODTypeToPrecisionType<int32_t>();
template <>
PrecisionType ConvertPODTypeToPrecisionType<uint32_t>();
template <>
PrecisionType ConvertPODTypeToPrecisionType<int64_t>();
template <>
PrecisionType ConvertPODTypeToPrecisionType<uint64_t>();
template <>
PrecisionType ConvertPODTypeToPrecisionType<float>();
template <>
PrecisionType ConvertPODTypeToPrecisionType<double>();
// Convert a NNAdapter precision code to a PaddleLite precision type
PrecisionType ConvertNNPrecisionCodeToPrecisionType(
    NNAdapterOperandPrecisionCode precision_code);

// Convert the PaddleLite dimensions to the NNAdapter dimensions
void ConvertDDimToNNDimensions(const DDim& input_dimensions,
                               int32_t* output_dimensions,
                               uint32_t* output_dimension_count = nullptr);
void ConvertVectorToNNDimensions(const std::vector<int64_t>& input_dimensions,
                                 int32_t* output_dimensions,
                                 uint32_t* output_dimension_count = nullptr);
DDim ConvertNNDimensionsToDDim(int32_t* input_dimensions,
                               uint32_t input_dimension_count);

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
