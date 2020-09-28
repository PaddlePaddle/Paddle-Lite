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

#include "lite/kernels/apu/bridges/graph.h"
#include <utility>
#include "lite/kernels/apu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace apu {

int Graph::Add(const std::string& name, std::shared_ptr<Node> node) {
  auto it = nodes_.find(name);

  if (it != nodes_.end()) {
    LOG(FATAL) << "[APU] Node" << name << " is redefined.";
    return -1;
  } else {
    VLOG(5) << " Add: " << name << " : " << node->index();
    auto ret = nodes_.insert(
        std::make_pair(name, std::vector<std::shared_ptr<Node>>()));
    CHECK(ret.second);
    it = ret.first;
  }
  operandIdx_ += 1;
  it->second.push_back(node);

  return it->second.size();
}

}  // namespace apu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
