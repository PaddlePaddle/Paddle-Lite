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

#include <memory>
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
// INTEL_OPENVINO_SELECT_DEVICE_NAMES=cpu,gpu or
// INTEL_OPENVINO_SELECT_DEVICE_NAMES=cpu
#define INTEL_OPENVINO_SELECT_DEVICE_NAMES "INTEL_OPENVINO_SELECT_DEVICE_NAMES"

// Convert NNAdapterAutoPadCode to OpenVINO ov::op::PadType
PadType ConvertToOVPadType(const NNAdapterAutoPadCode& auto_pad_code);
// Convert NNAdapterOperandPrecisionCode to OpenVINO ov::element::Type
ElementType ConvertToOVElementType(
    const NNAdapterOperandPrecisionCode& precision_code);
// Convert vector to OpenVINO ov::Shape
template <typename T>
Shape ConvertToOVShape(std::vector<T> dimensions) {
  std::vector<size_t> ov_shape;
  for (auto dim : dimensions) {
    ov_shape.push_back(dim);
  }
  return Shape(ov_shape);
}

// Convert C/C++ POD types to ElementType
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

// Add const ov::Node and return ov::Output<ov::Node>
template <typename T>
std::shared_ptr<OutputNode> AddConstOutputNode(std::vector<size_t> dimensions,
                                               std::vector<T> values) {
  auto constant_op = std::make_shared<default_opset::Constant>(
      GetElementType<T>(), Shape(dimensions), values);
  return std::make_shared<OutputNode>(constant_op->output(0));
}

}  // namespace intel_openvino
}  // namespace nnadapter
