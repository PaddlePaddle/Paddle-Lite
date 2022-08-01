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

#include "openvino/opsets/opset8.hpp"

#define NNADAPTER_INTEL_OPENVINO_VERSION_GREATER_EQUAL(major, minor, patch) \
  NNADAPTER_INTEL_OPENVINO_MAJOR_VERSION * 100 +                            \
          NNADAPTER_INTEL_OPENVINO_MINOR_VERSION * 10 +                     \
          NNADAPTER_INTEL_OPENVINO_PATCH_VERSION >=                         \
      major * 100 + minor * 10 + patch

#if NNADAPTER_INTEL_OPENVINO_VERSION_GREATER_EQUAL(2022, 2, 0)
#include "openvino/opsets/opset9.hpp"
#endif

namespace nnadapter {
namespace intel_openvino {

namespace default_opset = ov::opset8;
using Node = ov::Node;
using OutputNode = ov::Output<ov::Node>;
using PadType = ov::op::PadType;
using PadMode = ov::op::PadMode;
using ElementType = ov::element::Type;
using Shape = ov::Shape;

using Tensor = OutputNode;
using TensorVector = ov::OutputVector;
using Operator = Node;
}  // namespace intel_openvino
}  // namespace nnadapter
