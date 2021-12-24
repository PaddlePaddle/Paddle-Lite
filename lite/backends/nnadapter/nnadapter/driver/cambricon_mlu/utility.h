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

#include <cn_api.h>
#include <cnrt.h>
#include <interface_builder.h>
#include <interface_network.h>
#include <interface_runtime.h>
#include <memory>
#include <vector>
#include "core/hal/types.h"

namespace nnadapter {
namespace cambricon_mlu {

// The following environment variables can be used at runtime:
//
#define CAMBRICON_MLU_BUILD_CONFIG_FILE_PATH \
  "CAMBRICON_MLU_BUILD_CONFIG_FILE_PATH"

#define MLU_CNRT_CHECK(msg) \
  NNADAPTER_CHECK_EQ(msg, cnrtSuccess) << (msg) << " " << cnrtGetErrorStr(msg)

// Convert NNAdapter types to magicmind dtype
magicmind::DataType ConvertToMagicMindDtype(
    NNAdapterOperandPrecisionCode input_precision);
// Convert NNAdapter operand layout to magicmind layout
magicmind::Layout ConvertToMagicMindDataLayout(
    NNAdapterOperandLayoutCode input_layout);
// Convert NNAdapter operand layout to magicmind axis, if NHWC, axis = 3.
int64_t ConvertToMagicMindAxis(NNAdapterOperandLayoutCode input_layout);
// Convert NNAdapter dims to magicmind dims
magicmind::Dims ConvertToMagicMindDims(const int32_t* input_dimensions,
                                       uint32_t input_dimensions_count);
bool IsDeviceMemory(void* pointer);

template <typename T>
struct MMDestroyer {
  void operator()(T* t) {
    if (t) {
      t->Destroy();
    }
  }
};

template <typename T>
using MMUniquePtrType = std::unique_ptr<T, MMDestroyer<T>>;

}  // namespace cambricon_mlu
}  // namespace nnadapter
