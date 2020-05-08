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

#include <dlfcn.h>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace apu {

// Type/tensor converters for converting Paddle type/tensor to HiAI type/tensor
bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname);

void insert_transpose_node(void* ctx,
                           const std::string& input_name,
                           const std::string& output_name,
                           std::vector<uint32_t> input_shape,
                           std::vector<uint32_t> output_shape,
                           std::vector<int32_t> axis,
                           float scale,
                           int32_t zeroPoint);

void transpose(const int8_t* input_data,
               uint8_t* output_data,
               std::vector<uint32_t> input_shape,
               std::vector<uint32_t> axis);

void transposeAsym(const int8_t* input_data,
                   uint8_t* output_data,
                   std::vector<uint32_t> input_shape,
                   std::vector<uint32_t> axis);

void float2int32(const float* bias_data,
                 float input_scale,
                 std::vector<float> weight_scale,
                 int32_t* int32_bias_data);

}  // namespace apu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
