// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/intel_openvino/utility.h"
#include <map>
#include <memory>
#include <vector>

namespace nnadapter {
namespace intel_openvino {

void InitializeDeviceConfig(
    const std::string& device,
    std::shared_ptr<ov::Core> core,
    std::shared_ptr<std::map<std::string, ov::AnyMap>> config) {
  NNADAPTER_VLOG(5) << "Initialize Openvino device config.";
  core->set_property(device, config->at(device));
}

PadType ConvertToOVPadType(const NNAdapterAutoPadCode& auto_pad_code) {
  switch (auto_pad_code) {
    case NNADAPTER_AUTO_PAD_VALID:
      return PadType::VALID;
    case NNADAPTER_AUTO_PAD_SAME:
      return PadType::SAME_UPPER;
    case NNADAPTER_AUTO_PAD_NONE:
      return PadType::NOTSET;
    default:
      return PadType::NOTSET;
  }
}

Shape ConvertToOVShape(const NNAdapterOperandDimensionType& dimensions) {
  std::vector<size_t> ov_shape;
  int count = dimensions.count;
  auto data = dimensions.data;
  for (int i = 0; i < count; i++) {
    NNADAPTER_CHECK_NE(data[i], NNADAPTER_UNKNOWN);
    ov_shape.emplace_back(data[i]);
  }
  return Shape(ov_shape);
}

ov::PartialShape ConvertDynamicDimensions(
    NNAdapterOperandDimensionType* dimensions) {
  ov::PartialShape partial_shape;
  int count = dimensions->count;
  int dynamic_count = dimensions->dynamic_count;
  auto& dynamic_data = dimensions->dynamic_data;
  for (int i = 0; i < count; i++) {
    int min_shape = dynamic_data[0][i];
    int max_shape = dynamic_data[0][i];
    bool shape_joined = false;
    for (int j = 0; j < dynamic_count; j++) {
      // Number -1 has highest priority.
      int shape = dynamic_data[j][i];
      if (shape == -1) {
        partial_shape.push_back(ov::Dimension());
        shape_joined = true;
        break;
      }
      if (shape < min_shape) {
        min_shape = shape;
      }
      if (shape > max_shape) {
        max_shape = shape;
      }
    }
    if (shape_joined) continue;
    if (min_shape == max_shape) {
      partial_shape.push_back(ov::Dimension(min_shape));
    } else {
      partial_shape.push_back(ov::Dimension(min_shape, max_shape));
    }
  }
  return partial_shape;
}

ElementType ConvertToOVElementType(
    const NNAdapterOperandPrecisionCode& precision_code) {
  switch (precision_code) {
    case NNADAPTER_BOOL8:
      return ov::element::boolean;
    case NNADAPTER_INT8:
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
      return ov::element::i8;
    case NNADAPTER_UINT8:
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      return ov::element::u8;
    case NNADAPTER_INT16:
    case NNADAPTER_QUANT_INT16_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT16_SYMM_PER_CHANNEL:
      return ov::element::i16;
    case NNADAPTER_INT32:
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL:
      return ov::element::i32;
    case NNADAPTER_UINT32:
    case NNADAPTER_QUANT_UINT32_ASYMM_PER_LAYER:
      return ov::element::u32;
    case NNADAPTER_INT64:
      return ov::element::i64;
    case NNADAPTER_UINT64:
      return ov::element::u64;
    case NNADAPTER_FLOAT16:
      return ov::element::f16;
    case NNADAPTER_FLOAT32:
      return ov::element::f32;
    case NNADAPTER_FLOAT64:
      return ov::element::f64;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(precision_code)
          << ") to OpenVINO element type !";
  }
  return ov::element::f32;
}

PadMode ConvertPadModeCodeToOVPadMode(int pad_mode_code) {
  switch (pad_mode_code) {
    case NNADAPTER_PAD_MODE_CONSTANT:
      return PadMode::CONSTANT;
    case NNADAPTER_PAD_MODE_REFLECT:
      return PadMode::REFLECT;
    case NNADAPTER_PAD_MODE_EDGE:
    case NNADAPTER_PAD_MODE_REPLICATE:
      return PadMode::EDGE;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand pad mode code("
          << pad_mode_code << ") to pad mode !";
      break;
  }
  return PadMode::CONSTANT;
}

template <>
ElementType GetElementType<int8_t>() {
  return ov::element::i8;
}
template <>
ElementType GetElementType<int16_t>() {
  return ov::element::i16;
}
template <>
ElementType GetElementType<int32_t>() {
  return ov::element::i32;
}
template <>
ElementType GetElementType<int64_t>() {
  return ov::element::i64;
}
template <>
ElementType GetElementType<uint8_t>() {
  return ov::element::u8;
}
template <>
ElementType GetElementType<uint16_t>() {
  return ov::element::u16;
}
template <>
ElementType GetElementType<uint32_t>() {
  return ov::element::u32;
}
template <>
ElementType GetElementType<uint64_t>() {
  return ov::element::u64;
}
template <>
ElementType GetElementType<float>() {
  return ov::element::f32;
}
template <>
ElementType GetElementType<double>() {
  return ov::element::f64;
}

}  // namespace intel_openvino
}  // namespace nnadapter
