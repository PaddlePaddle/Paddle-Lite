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

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "core/types.h"
#include "driver/intel_openvino/default_opset.h"
#include "openvino/openvino.hpp"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/string.h"

namespace nnadapter {
namespace intel_openvino {

// The following environment variables can be used at runtime:
// Specify the list of device names, such as
// INTEL_OPENVINO_SELECT_DEVICE_NAMES=CPU,GPU or
// INTEL_OPENVINO_SELECT_DEVICE_NAMES=CPU
#define INTEL_OPENVINO_SELECT_DEVICE_NAMES "INTEL_OPENVINO_SELECT_DEVICE_NAMES"

// Specify inference thread numbers.
#define INTEL_OPENVINO_INFERENCE_NUM_THREADS \
  "INTEL_OPENVINO_INFERENCE_NUM_THREADS"

// Initialize device config for runtime core.
void InitializeDeviceConfig(
    const std::string& device,
    std::shared_ptr<ov::Core> core,
    std::shared_ptr<std::map<std::string, ov::AnyMap>> config);

// Convert NNAdapterAutoPadCode to OpenVINO ov::op::PadType
PadType ConvertToOVPadType(const NNAdapterAutoPadCode& auto_pad_code);
// Convert NNAdapterPadModeCode to OpenVINO ov::op::PadMode
PadMode ConvertPadModeCodeToOVPadMode(int pad_mode_code);
// Convert NNAdapterOperandPrecisionCode to OpenVINO ov::element::Type
ElementType ConvertToOVElementType(
    const NNAdapterOperandPrecisionCode& precision_code);
// Convert vector to OpenVINO ov::Shape
template <typename T>
Shape ConvertToOVShape(std::vector<T> dimensions) {
  std::vector<size_t> ov_shape;
  for (auto dim : dimensions) {
    if (dim == NNADAPTER_UNKNOWN) {
      ov_shape.push_back(-1);
    } else {
      ov_shape.push_back(dim);
    }
  }
  return Shape(ov_shape);
}

// Convert NNadapterOperandDimensionType to ov::Shape.
Shape ConvertToOVShape(const NNAdapterOperandDimensionType& dimensions);

// Collect dynamic shape info and convert to ov::PartialShape.
ov::PartialShape ConvertDynamicDimensions(
    NNAdapterOperandDimensionType* dimensions);

// Convert C/C++ POD types to OpenVINO ov::element::Type
template <typename T>
ElementType GetElementType() {
  NNADAPTER_LOG(FATAL) << "Unable to convert " << typeid(T).name()
                       << " to ElementType";
  return ov::element::f32;
}
template <>
ElementType GetElementType<int8_t>();
template <>
ElementType GetElementType<int16_t>();
template <>
ElementType GetElementType<int32_t>();
template <>
ElementType GetElementType<int64_t>();
template <>
ElementType GetElementType<uint8_t>();
template <>
ElementType GetElementType<uint16_t>();
template <>
ElementType GetElementType<uint32_t>();
template <>
ElementType GetElementType<uint64_t>();
template <>
ElementType GetElementType<float>();
template <>
ElementType GetElementType<double>();

}  // namespace intel_openvino
}  // namespace nnadapter
