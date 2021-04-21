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

#include "lite/kernels/nnadapter/bridges/utility.h"
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

bool hasInput(const OpInfo* op_info,
              const Scope* scope,
              const std::string& arg_name) {
  return op_info->HasInput(arg_name) && op_info->Input(arg_name).size() > 0 &&
         scope->FindVar(op_info->Input(arg_name).front());
}

bool hasOutput(const OpInfo* op_info,
               const Scope* scope,
               const std::string& arg_name) {
  return op_info->HasOutput(arg_name) && op_info->Output(arg_name).size() > 0 &&
         scope->FindVar(op_info->Output(arg_name).front());
}

bool isPerChannelScales(const std::vector<float>& scales) {
  const float threshold = 1e-5f;
  size_t size = scales.size();
  CHECK_GT(size, 0) << "The size of scales should be greater than 0.";
  auto ref_scale = scales[0];
  for (size_t i = 1; i < size; i++) {
    auto cur_scale = scales[i];
    if (fabs(cur_scale - ref_scale) > threshold) {
      return false;
    }
  }
  return true;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
