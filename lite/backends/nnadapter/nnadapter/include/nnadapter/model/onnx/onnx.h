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

#include <vector>
#include "core/types.h"

namespace nnadapter {
namespace model {
namespace onnx {

// Convert core::Model to ONNX model and serialize it to buffer or file
bool Serialize(core::Model* model,
               std::vector<uint8_t>* buffer,
               int32_t opset_version = 11);
bool Serialize(core::Model* model,
               const char* path,
               int32_t opset_version = 11);
}  // namespace onnx
}  // namespace model
}  // namespace nnadapter
