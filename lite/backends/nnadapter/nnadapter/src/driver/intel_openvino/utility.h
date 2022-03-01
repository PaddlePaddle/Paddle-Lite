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
// Add const ov::Node and return ov::Output<ov::Node>
template <typename T>
std::shared_ptr<OutputNode> AddConstOutputNode(std::vector<size_t> dimensions,
                                               std::vector<T> values) {
  NNADAPTER_LOG(FATAL) << "Unable to add const output node with type "
                       << typeid(T).name();
  return nullptr;
}
#define FUNCTION_ADD_CONST_OUTPUT_NODE_DECLARE(type, element_type) \
  template <>                                                      \
  std::shared_ptr<OutputNode> AddConstOutputNode<type>(            \
      std::vector<size_t> dimensions, std::vector<type> values);
FUNCTION_ADD_CONST_OUTPUT_NODE_DECLARE(int8_t, ov::element::i8)
FUNCTION_ADD_CONST_OUTPUT_NODE_DECLARE(int16_t, ov::element::i16)
FUNCTION_ADD_CONST_OUTPUT_NODE_DECLARE(int32_t, ov::element::i32)
FUNCTION_ADD_CONST_OUTPUT_NODE_DECLARE(int64_t, ov::element::i64)
FUNCTION_ADD_CONST_OUTPUT_NODE_DECLARE(uint8_t, ov::element::u8)
FUNCTION_ADD_CONST_OUTPUT_NODE_DECLARE(uint16_t, ov::element::u16)
FUNCTION_ADD_CONST_OUTPUT_NODE_DECLARE(uint32_t, ov::element::u32)
FUNCTION_ADD_CONST_OUTPUT_NODE_DECLARE(uint64_t, ov::element::u64)
FUNCTION_ADD_CONST_OUTPUT_NODE_DECLARE(float, ov::element::f32)
FUNCTION_ADD_CONST_OUTPUT_NODE_DECLARE(double, ov::element::f64)
#undef FUNCTION_ADD_CONST_OUTPUT_NODE_DECLARE

}  // namespace intel_openvino
}  // namespace nnadapter
