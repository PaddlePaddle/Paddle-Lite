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

#include "lite/backends/xpu/builder.h"
#include <mutex>  // NOLINT
#include <utility>
#include "lite/backends/xpu/runtime.h"

namespace paddle {
namespace lite {
namespace xpu {

std::string UniqueName(const std::string& prefix) {
  static std::mutex counter_mtx;
  static std::unordered_map<std::string, int> counter_map;
  std::unique_lock<std::mutex> counter_lck(counter_mtx);
  int counter = 1;
  auto it = counter_map.find(prefix);
  if (it == counter_map.end()) {
    counter_map[prefix] = counter;
  } else {
    counter = ++(it->second);
  }
  return prefix + "_" + std::to_string(counter);
}

// Build IR graph to model, and store model data into lite tensor
bool BuildModel(std::vector<xtcl::xExpr>& inputs,   // NOLINT
                std::vector<xtcl::xExpr>& outputs,  // NOLINT
                lite::Tensor* model_data) {
  LOG(INFO) << "[XPU] Build model.";
  CHECK_GT(inputs.size(), 0);
  CHECK_GT(outputs.size(), 0);
  CHECK_NE(model_data, 0);
  return true;
}

}  // namespace xpu
}  // namespace lite
}  // namespace paddle
