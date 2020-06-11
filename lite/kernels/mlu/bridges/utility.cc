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

#include "lite/kernels/mlu/bridges/utility.h"

#include <utility>

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

void transpose2d(float* input_data,
                 float* output_data,
                 std::vector<int> input_shape) {
  CHECK_EQ(input_shape.size(), 2);
  int old_index = -1;
  int new_index = -1;
  for (int i = 0; i < input_shape[0]; i++) {
    for (int j = 0; j < input_shape[1]; j++) {
      old_index = i * input_shape[1] + j;
      new_index = j * input_shape[0] + i;
      output_data[new_index] = input_data[old_index];
    }
  }
}

void dequant(float* dst, int8_t* src, size_t size, float scale) {
  for (size_t i = 0; i < size; ++i) {
    dst[i] = static_cast<float>(src[i]) * scale;
  }
}

void dequant(float* dst,
             int8_t* src,
             size_t size_o,
             size_t size,
             size_t size_in,
             std::vector<float> scales) {
  for (size_t out = 0; out < size_o; ++out) {
    for (size_t s = 0; s < size; ++s) {
      auto scale = scales[s];
      for (size_t in = 0; in < size_in; ++in) {
        int idx = in + s * size_in + out * size_in * size;
        dst[idx] = static_cast<float>(src[idx]) * scale;
      }
    }
  }
}

cnmlActiveFunction_t OpTypeToCNMLActType(std::string op_type) {
  if (op_type == "relu") {
    return CNML_ACTIVE_RELU;
  } else if (op_type == "sigmoid") {
    return CNML_ACTIVE_SIGMOID;
  } else if (op_type == "tanh") {
    return CNML_ACTIVE_TANH;
  } else if (op_type == "relu1") {
    return CNML_ACTIVE_RELU1;
  } else if (op_type == "relu6") {
    return CNML_ACTIVE_RELU6;
  } else if (op_type == "hard_sigmoid") {
    return CNML_ACTIVE_HARD_SIGMOID;
  }
  LOG(FATAL) << "CNML Unspoorted op type " << op_type;
  return CNML_ACTIVE_NONE;
}

bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname) {
  auto iarg_names = op_info->input_argnames();
  if (std::find(iarg_names.begin(), iarg_names.end(), argname) !=
      iarg_names.end()) {
    auto inputs = op_info->Input(argname);
    if (inputs.empty()) {
      return false;
    }
    auto var_name = inputs.front();
    auto var = scope->FindVar(var_name);
    return var != nullptr;
  } else {
    return false;
  }
}
}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
