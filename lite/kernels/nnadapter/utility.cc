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

#include "lite/kernels/nnadapter/utility.h"
#include <math.h>

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

bool HasInput(const OpInfo* op_info,
              const Scope* scope,
              const std::string& arg_name) {
  return op_info->HasInput(arg_name) && op_info->Input(arg_name).size() > 0 &&
         scope->FindVar(op_info->Input(arg_name).front());
}

bool HasOutput(const OpInfo* op_info,
               const Scope* scope,
               const std::string& arg_name) {
  return op_info->HasOutput(arg_name) && op_info->Output(arg_name).size() > 0 &&
         scope->FindVar(op_info->Output(arg_name).front());
}

bool IsValidSymmQuantParams(const std::vector<float>& quant_scales,
                            uint32_t quant_channel_dim) {
  auto quant_scale_count = quant_scales.size();
  if (quant_scale_count == 0) return false;
  CHECK(quant_channel_dim < NNADAPTER_MAX_SIZE_OF_DIMENSIONS);
  for (size_t i = 0; i < quant_scale_count; i++) {
    if (quant_scales[i] < 0.f) return false;
  }
  return true;
}

bool IsValidSymmPerLayerQuantParams(const std::vector<float>& quant_scales) {
  if (quant_scales.size() != 1) return false;
  return quant_scales[0] >= 0.f;
}

bool IsValidSymmPerChannelQuantParams(const std::vector<float>& quant_scales,
                                      uint32_t quant_channel_dim,
                                      float threshold) {
  auto quant_scale_count = quant_scales.size();
  if (quant_scale_count <= 1) return false;
  CHECK(quant_channel_dim < NNADAPTER_MAX_SIZE_OF_DIMENSIONS);
  auto ref_quant_scale = quant_scales[0];
  if (ref_quant_scale < 0.f) return false;
  for (size_t i = 1; i < quant_scale_count; i++) {
    auto cur_quant_scale = quant_scales[i];
    if (cur_quant_scale < 0.f) return false;
    if (fabs(cur_quant_scale - ref_quant_scale) > threshold) {
      return true;
    }
  }
  return false;
}

bool IsSameSymmQuantParams(const std::vector<float>& target_quant_scales,
                           const std::vector<float>& ref_quant_scales,
                           int32_t target_quant_channel_dim,
                           int32_t ref_quant_channel_dim,
                           float threshold) {
  CHECK(IsValidSymmQuantParams(target_quant_scales, target_quant_channel_dim));
  CHECK(IsValidSymmQuantParams(ref_quant_scales, ref_quant_channel_dim));
  auto quant_scale_count = target_quant_scales.size();
  if (quant_scale_count == ref_quant_scales.size()) {
    for (size_t i = 0; i < quant_scale_count; i++) {
      if (fabs(target_quant_scales[i] - ref_quant_scales[i]) > threshold)
        return false;
    }
    return target_quant_channel_dim == ref_quant_channel_dim;
  }
  return false;
}

int64_t ProductionOfDimensions(const int32_t* input_dimensions,
                               uint32_t input_dimension_count) {
  int64_t production = 1;
  for (uint32_t i = 0; i < input_dimension_count; i++) {
    auto dimension = input_dimensions[i];
    CHECK_GT(dimension, 0);
    production *= dimension;
  }
  return production;
}

int64_t ProductionOfDimensions(const std::vector<int32_t>& input_dimensions) {
  return !input_dimensions.empty()
             ? ProductionOfDimensions(&input_dimensions[0],
                                      input_dimensions.size())
             : 1;
}

bool IsNNSymmQuantType(NNAdapterOperandPrecisionCode precision_code) {
  return IsNNSymmPerLayerQuantType(precision_code) ||
         IsNNSymmPerChannelQuantType(precision_code);
}

bool IsNNSymmPerLayerQuantType(NNAdapterOperandPrecisionCode precision_code) {
  return IsNNInt8SymmPerLayerQuantType(precision_code) ||
         IsNNInt16SymmPerLayerQuantType(precision_code) ||
         IsNNInt32SymmPerLayerQuantType(precision_code);
}

bool IsNNSymmPerChannelQuantType(NNAdapterOperandPrecisionCode precision_code) {
  return IsNNInt8SymmPerChannelQuantType(precision_code) ||
         IsNNInt16SymmPerChannelQuantType(precision_code) ||
         IsNNInt32SymmPerChannelQuantType(precision_code);
}

bool IsNNInt8SymmQuantType(NNAdapterOperandPrecisionCode precision_code) {
  return IsNNInt8SymmPerLayerQuantType(precision_code) ||
         IsNNInt8SymmPerChannelQuantType(precision_code);
}

bool IsNNInt8SymmPerLayerQuantType(
    NNAdapterOperandPrecisionCode precision_code) {
  return precision_code == NNADAPTER_QUANT_INT8_SYMM_PER_LAYER;
}

bool IsNNInt8SymmPerChannelQuantType(
    NNAdapterOperandPrecisionCode precision_code) {
  return precision_code == NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL;
}

bool IsNNInt16SymmQuantType(NNAdapterOperandPrecisionCode precision_code) {
  return IsNNInt16SymmPerLayerQuantType(precision_code) ||
         IsNNInt16SymmPerChannelQuantType(precision_code);
}

bool IsNNInt16SymmPerLayerQuantType(
    NNAdapterOperandPrecisionCode precision_code) {
  return precision_code == NNADAPTER_QUANT_INT16_SYMM_PER_LAYER;
}

bool IsNNInt16SymmPerChannelQuantType(
    NNAdapterOperandPrecisionCode precision_code) {
  return precision_code == NNADAPTER_QUANT_INT16_SYMM_PER_CHANNEL;
}

bool IsNNInt32SymmQuantType(NNAdapterOperandPrecisionCode precision_code) {
  return IsNNInt32SymmPerLayerQuantType(precision_code) ||
         IsNNInt32SymmPerChannelQuantType(precision_code);
}

bool IsNNInt32SymmPerLayerQuantType(
    NNAdapterOperandPrecisionCode precision_code) {
  return precision_code == NNADAPTER_QUANT_INT32_SYMM_PER_LAYER;
}

bool IsNNInt32SymmPerChannelQuantType(
    NNAdapterOperandPrecisionCode precision_code) {
  return precision_code == NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL;
}

bool IsNNSymmQuantType(const NNAdapterOperandType& operand_type) {
  return IsNNSymmQuantType(operand_type.precision);
}

bool IsNNSymmPerLayerQuantType(const NNAdapterOperandType& operand_type) {
  return IsNNSymmPerLayerQuantType(operand_type.precision);
}

bool IsNNSymmPerChannelQuantType(const NNAdapterOperandType& operand_type) {
  return IsNNSymmPerChannelQuantType(operand_type.precision);
}

bool IsNNInt8SymmQuantType(const NNAdapterOperandType& operand_type) {
  return IsNNInt8SymmQuantType(operand_type.precision);
}

bool IsNNInt8SymmPerLayerQuantType(const NNAdapterOperandType& operand_type) {
  return IsNNInt8SymmPerLayerQuantType(operand_type.precision);
}

bool IsNNInt8SymmPerChannelQuantType(const NNAdapterOperandType& operand_type) {
  return IsNNInt8SymmPerChannelQuantType(operand_type.precision);
}

bool IsNNInt16SymmQuantType(const NNAdapterOperandType& operand_type) {
  return IsNNInt16SymmQuantType(operand_type.precision);
}

bool IsNNInt16SymmPerLayerQuantType(const NNAdapterOperandType& operand_type) {
  return IsNNInt16SymmPerLayerQuantType(operand_type.precision);
}

bool IsNNInt16SymmPerChannelQuantType(
    const NNAdapterOperandType& operand_type) {
  return IsNNInt16SymmPerChannelQuantType(operand_type.precision);
}

bool IsNNInt32SymmQuantType(const NNAdapterOperandType& operand_type) {
  return IsNNInt32SymmQuantType(operand_type.precision);
}

bool IsNNInt32SymmPerLayerQuantType(const NNAdapterOperandType& operand_type) {
  return IsNNInt32SymmPerLayerQuantType(operand_type.precision);
}

bool IsNNInt32SymmPerChannelQuantType(
    const NNAdapterOperandType& operand_type) {
  return IsNNInt32SymmPerChannelQuantType(operand_type.precision);
}

bool GetNNSymmQuantParams(const NNAdapterOperandType& operand_type,
                          std::vector<float>* quant_scales,
                          uint32_t* quant_channel_dim) {
  if (IsNNSymmPerLayerQuantType(operand_type.precision)) {
    *quant_scales = {operand_type.symm_per_layer_params.scale};
    return true;
  } else if (IsNNSymmPerChannelQuantType(operand_type.precision)) {
    auto quant_scale_count = operand_type.symm_per_channel_params.scale_count;
    quant_scales->resize(quant_scale_count);
    memcpy(quant_scales->data(),
           operand_type.symm_per_channel_params.scales,
           quant_scale_count * sizeof(float));
    if (quant_channel_dim) {
      *quant_channel_dim = operand_type.symm_per_channel_params.channel_dim;
    }
    return true;
  }
  return false;
}

bool SetNNSymmQuantParams(NNAdapterOperandType* operand_type,
                          const std::vector<float>& quant_scales,
                          uint32_t quant_channel_dim) {
  auto quant_scale_count = quant_scales.size();
  if (IsNNSymmPerLayerQuantType(operand_type->precision) &&
      quant_scale_count == 1) {
    operand_type->symm_per_layer_params.scale = quant_scales[0];
    return true;
  } else if (IsNNSymmPerChannelQuantType(operand_type->precision) &&
             quant_scale_count > 1) {
    memcpy(operand_type->symm_per_channel_params.scales,
           quant_scales.data(),
           quant_scale_count * sizeof(float));
    operand_type->symm_per_channel_params.channel_dim = quant_channel_dim;
    return true;
  }
  return false;
}

int64_t GetNNOperandPrecisionDataLength(
    NNAdapterOperandPrecisionCode precision_code) {
  switch (precision_code) {
    case NNADAPTER_BOOL8:
    case NNADAPTER_INT8:
    case NNADAPTER_UINT8:
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      return 1;
    case NNADAPTER_INT16:
    case NNADAPTER_UINT16:
    case NNADAPTER_FLOAT16:
    case NNADAPTER_QUANT_INT16_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT16_SYMM_PER_CHANNEL:
    case NNADAPTER_QUANT_UINT16_ASYMM_PER_LAYER:
      return 2;
    case NNADAPTER_INT32:
    case NNADAPTER_UINT32:
    case NNADAPTER_FLOAT32:
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL:
    case NNADAPTER_QUANT_UINT32_ASYMM_PER_LAYER:
      return 4;
    case NNADAPTER_INT64:
    case NNADAPTER_UINT64:
    case NNADAPTER_FLOAT64:
      return 8;
    default:
      LOG(FATAL) << "Unable to get the length of precision code("
                 << static_cast<int>(precision_code) << ").";
      break;
  }
  return 0;
}

int64_t GetNNOperandPrecisionDataLength(
    const NNAdapterOperandType& operand_type) {
  return GetNNOperandPrecisionDataLength(operand_type.precision);
}

int64_t GetNNOperandTypeBufferLength(const NNAdapterOperandType& operand_type) {
  auto production = ProductionOfDimensions(operand_type.dimensions.data,
                                           operand_type.dimensions.count);
  return GetNNOperandPrecisionDataLength(operand_type.precision) * production;
}

template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<bool>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim) {
  CHECK_EQ(quant_scale_count, 0);
  return NNADAPTER_BOOL8;
}

template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<int8_t>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim) {
  if (quant_scale_count > 0) {
    CHECK(quant_scales);
    // INT8 only supports symmetric per-layer or per-channel quantization
    return quant_scale_count > 1 ? NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL
                                 : NNADAPTER_QUANT_INT8_SYMM_PER_LAYER;
  }
  return NNADAPTER_INT8;
}

template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<uint8_t>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim) {
  CHECK_EQ(quant_scale_count, 0);
  return NNADAPTER_UINT8;
}

template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<int16_t>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim) {
  if (quant_scale_count > 0) {
    CHECK(quant_scales);
    // INT8 only supports symmetric per-layer or per-channel quantization
    return quant_scale_count > 1 ? NNADAPTER_QUANT_INT16_SYMM_PER_CHANNEL
                                 : NNADAPTER_QUANT_INT16_SYMM_PER_LAYER;
  }
  return NNADAPTER_INT16;
}

template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<uint16_t>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim) {
  CHECK_EQ(quant_scale_count, 0);
  return NNADAPTER_UINT16;
}

template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<int32_t>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim) {
  if (quant_scale_count > 0) {
    CHECK(quant_scales);
    // INT32 only supports symmetric per-layer or per-channel quantization
    return quant_scale_count > 1 ? NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL
                                 : NNADAPTER_QUANT_INT32_SYMM_PER_LAYER;
  }
  return NNADAPTER_INT32;
}

template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<uint32_t>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim) {
  CHECK_EQ(quant_scale_count, 0);
  return NNADAPTER_UINT32;
}

template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<int64_t>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim) {
  CHECK_EQ(quant_scale_count, 0);
  return NNADAPTER_INT64;
}

template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<uint64_t>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim) {
  CHECK_EQ(quant_scale_count, 0);
  return NNADAPTER_UINT64;
}

template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<float>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim) {
  CHECK_EQ(quant_scale_count, 0);
  return NNADAPTER_FLOAT32;
}

template <>
NNAdapterOperandPrecisionCode ConvertPODTypeToNNPrecisionCode<double>(
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim) {
  CHECK_EQ(quant_scale_count, 0);
  return NNADAPTER_FLOAT64;
}

NNAdapterOperandPrecisionCode ConvertFluidDataTypeToNNPrecisionCode(
    int fluid_dtype,
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim) {
  if (fluid_dtype == 21) {  // INT8 = 21
    if (quant_scale_count > 0) {
      CHECK(quant_scales);
      // INT8 only supports symmetric per-layer or per-channel quantization
      return quant_scale_count > 1 ? NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL
                                   : NNADAPTER_QUANT_INT8_SYMM_PER_LAYER;
    }
    return NNADAPTER_INT8;
  } else if (fluid_dtype == 1) {  // INT16 = 1
    if (quant_scale_count > 0) {
      CHECK(quant_scales);
      // INT16 only supports symmetric per-layer or per-channel quantization
      return quant_scale_count > 1 ? NNADAPTER_QUANT_INT16_SYMM_PER_CHANNEL
                                   : NNADAPTER_QUANT_INT16_SYMM_PER_LAYER;
    }
    return NNADAPTER_INT16;
  } else if (fluid_dtype == 2) {  // INT32 = 2
    if (quant_scale_count > 0) {
      CHECK(quant_scales);
      // INT32 only supports symmetric per-layer or per-channel quantization
      return quant_scale_count > 1 ? NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL
                                   : NNADAPTER_QUANT_INT32_SYMM_PER_LAYER;
    }
    return NNADAPTER_INT32;
  }
  CHECK_EQ(quant_scale_count, 0);
  switch (fluid_dtype) {
    case 0:  // BOOL = 0;
      return NNADAPTER_BOOL8;
    case 2:  // INT32 = 2
      return NNADAPTER_INT32;
    case 3:  // INT64 = 3
      return NNADAPTER_INT64;
    case 4:  // FP16 = 4
      return NNADAPTER_FLOAT16;
    case 5:  // FP32 = 5
      return NNADAPTER_FLOAT32;
    case 6:  // FP64 = 6
      return NNADAPTER_FLOAT64;
    case 20:  // UINT8 = 20
      return NNADAPTER_UINT8;
    default:
      LOG(FATAL) << "Unable to convert a fluid data type(" << fluid_dtype
                 << ") to a NNAdapter precision code";
      break;
  }
  return NNADAPTER_FLOAT32;
}

NNAdapterOperandPrecisionCode ConvertPrecisionTypeToNNPrecisionCode(
    PrecisionType precision_type,
    const float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim) {
  if (precision_type == PRECISION(kInt8)) {
    if (quant_scale_count > 0) {
      CHECK(quant_scales);
      // INT8 only supports symmetric per-layer or per-channel quantization
      return quant_scale_count > 1 ? NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL
                                   : NNADAPTER_QUANT_INT8_SYMM_PER_LAYER;
    }
    return NNADAPTER_INT8;
  } else if (precision_type == PRECISION(kInt16)) {
    if (quant_scale_count > 0) {
      CHECK(quant_scales);
      // INT16 only supports symmetric per-layer or per-channel quantization
      return quant_scale_count > 1 ? NNADAPTER_QUANT_INT16_SYMM_PER_CHANNEL
                                   : NNADAPTER_QUANT_INT16_SYMM_PER_LAYER;
    }
    return NNADAPTER_INT32;
  } else if (precision_type == PRECISION(kInt32)) {
    if (quant_scale_count > 0) {
      CHECK(quant_scales);
      // INT32 only supports symmetric per-layer or per-channel quantization
      return quant_scale_count > 1 ? NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL
                                   : NNADAPTER_QUANT_INT32_SYMM_PER_LAYER;
    }
    return NNADAPTER_INT32;
  }
  CHECK_EQ(quant_scale_count, 0);
  switch (precision_type) {
    case PRECISION(kFloat):
      return NNADAPTER_FLOAT32;
    case PRECISION(kFP16):
      return NNADAPTER_FLOAT16;
    case PRECISION(kBool):
      return NNADAPTER_BOOL8;
    case PRECISION(kInt64):
      return NNADAPTER_INT64;
    case PRECISION(kUInt8):
      return NNADAPTER_UINT8;
    case PRECISION(kFP64):
      return NNADAPTER_FLOAT64;
    default:
      LOG(FATAL) << "Unable to convert a precision type("
                 << lite_api::PrecisionToStr(precision_type)
                 << ") to a NNAdapter precision code!";
      break;
  }
  return NNADAPTER_FLOAT32;
}

NNAdapterOperationType ConvertUnaryActTypeToNNOperationType(
    const std::string& unary_act_op_type) {
  NNAdapterOperationType unary_act_op_code = NNADAPTER_UNKNOWN;
  if (unary_act_op_type == "sigmoid") {
    unary_act_op_code = NNADAPTER_SIGMOID;
  } else if (unary_act_op_type == "relu") {
    unary_act_op_code = NNADAPTER_RELU;
  } else if (unary_act_op_type == "relu6") {
    unary_act_op_code = NNADAPTER_RELU6;
  } else if (unary_act_op_type == "tanh") {
    unary_act_op_code = NNADAPTER_TANH;
  } else if (unary_act_op_type == "log") {
    unary_act_op_code = NNADAPTER_LOG;
  } else if (unary_act_op_type == "abs") {
    unary_act_op_code = NNADAPTER_ABS;
  } else if (unary_act_op_type == "exp") {
    unary_act_op_code = NNADAPTER_EXP;
  } else if (unary_act_op_type == "floor") {
    unary_act_op_code = NNADAPTER_FLOOR;
  } else if (unary_act_op_type == "square") {
    unary_act_op_code = NNADAPTER_SQUARE;
  } else {
    LOG(WARNING) << "Unable to convert a unary activation type("
                 << unary_act_op_type << ") to a NNAdapter operation type!";
  }
  return unary_act_op_code;
}

NNAdapterAutoPadCode ConvertPaddingAlgorithmToNNAutoPadCode(
    const std::string& padding_algorithm) {
  NNAdapterAutoPadCode auto_pad_code;
  if (padding_algorithm == "EXPLICIT" || padding_algorithm.empty()) {
    auto_pad_code = NNADAPTER_AUTO_PAD_NONE;
  } else if (padding_algorithm == "SAME") {
    auto_pad_code = NNADAPTER_AUTO_PAD_SAME;
  } else if (padding_algorithm == "VALID") {
    auto_pad_code = NNADAPTER_AUTO_PAD_VALID;
  } else {
    LOG(FATAL) << "Unsupported padding_algorithm: " << padding_algorithm;
  }
  return auto_pad_code;
}

NNAdapterPadModeCode ConvertPadModeToNNPadModeCode(std::string mode) {
  if (mode == "constant" || mode == "zeros") {
    return NNADAPTER_PAD_MODE_CONSTANT;
  }
  if (mode == "reflect") {
    return NNADAPTER_PAD_MODE_REFLECT;
  }
  if (mode == "replicate") {
    return NNADAPTER_PAD_MODE_REPLICATE;
  }
  if (mode == "edge" || mode == "border") {
    return NNADAPTER_PAD_MODE_EDGE;
  }
  LOG(WARNING) << "Unsupported mode type: " << mode;
  return NNADAPTER_PAD_MODE_NONE;
}

NNAdapterInterpolateModeCode ConvertInterpolateModeToNNInterpolateModeCode(
    std::string mode) {
  if (mode == "bilinear") {
    return NNADAPTER_INTERPOLATE_MODE_BILINEAR;
  }
  if (mode == "nearest") {
    return NNADAPTER_INTERPOLATE_MODE_NEAREST;
  }
  LOG(WARNING) << "Unsupported mode type: " << mode;
  return NNADAPTER_INTERPOLATE_MODE_NONE;
}

template <>
PrecisionType ConvertPODTypeToPrecisionType<bool>() {
  return PRECISION(kBool);
}

template <>
PrecisionType ConvertPODTypeToPrecisionType<int8_t>() {
  return PRECISION(kInt8);
}

template <>
PrecisionType ConvertPODTypeToPrecisionType<uint8_t>() {
  return PRECISION(kUInt8);
}

template <>
PrecisionType ConvertPODTypeToPrecisionType<int16_t>() {
  return PRECISION(kInt16);
}

template <>
PrecisionType ConvertPODTypeToPrecisionType<int32_t>() {
  return PRECISION(kInt32);
}

template <>
PrecisionType ConvertPODTypeToPrecisionType<int64_t>() {
  return PRECISION(kInt64);
}

template <>
PrecisionType ConvertPODTypeToPrecisionType<float>() {
  return PRECISION(kFloat);
}

template <>
PrecisionType ConvertPODTypeToPrecisionType<double>() {
  return PRECISION(kFP64);
}

PrecisionType ConvertNNPrecisionCodeToPrecisionType(
    NNAdapterOperandPrecisionCode precision_code) {
  PrecisionType precision_type = PRECISION(kUnk);
  switch (precision_code) {
    case NNADAPTER_BOOL8:
      precision_type = PRECISION(kBool);
      break;
    case NNADAPTER_INT8:
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      precision_type = PRECISION(kInt8);
      break;
    case NNADAPTER_INT16:
    case NNADAPTER_QUANT_INT16_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT16_SYMM_PER_CHANNEL:
    case NNADAPTER_QUANT_UINT16_ASYMM_PER_LAYER:
      precision_type = PRECISION(kInt16);
      break;
    case NNADAPTER_INT32:
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL:
    case NNADAPTER_QUANT_UINT32_ASYMM_PER_LAYER:
      precision_type = PRECISION(kInt32);
      break;
    case NNADAPTER_INT64:
      precision_type = PRECISION(kInt64);
      break;
    case NNADAPTER_FLOAT16:
      precision_type = PRECISION(kFP16);
      break;
    case NNADAPTER_FLOAT32:
      precision_type = PRECISION(kFloat);
      break;
    case NNADAPTER_FLOAT64:
      precision_type = PRECISION(kFP64);
      break;
    default:
      LOG(FATAL) << "Unable to convert a NNAdapter precision code("
                 << static_cast<int>(precision_code) << ") to a precision type";
      break;
  }
  return precision_type;
}

void ConvertDDimToNNDimensions(const DDim& input_dimensions,
                               int32_t* output_dimensions,
                               uint32_t* output_dimension_count) {
  CHECK(output_dimensions);
  if (output_dimension_count) {
    *output_dimension_count = input_dimensions.size();
  }
  for (size_t i = 0; i < input_dimensions.size(); i++) {
    output_dimensions[i] = static_cast<int32_t>(input_dimensions[i]);
  }
}

void ConvertVectorToNNDimensions(const std::vector<int64_t>& input_dimensions,
                                 int32_t* output_dimensions,
                                 uint32_t* output_dimension_count) {
  CHECK(output_dimensions);
  if (output_dimension_count) {
    *output_dimension_count = input_dimensions.size();
  }
  for (size_t i = 0; i < input_dimensions.size(); i++) {
    output_dimensions[i] = static_cast<int32_t>(input_dimensions[i]);
  }
}

DDim ConvertNNDimensionsToDDim(int32_t* input_dimensions,
                               uint32_t input_dimension_count) {
  CHECK(input_dimensions);
  std::vector<int64_t> output_dimensions(input_dimension_count);
  for (int i = 0; i < input_dimension_count; i++) {
    output_dimensions[i] = static_cast<int64_t>(input_dimensions[i]);
  }
  return DDim(output_dimensions);
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
