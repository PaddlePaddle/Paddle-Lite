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

#include "lite/core/subgraph_bridge_registry.h"
#include <utility>

namespace paddle {
namespace lite {
namespace subgraph {

SubgraphBridgeRegistry& SubgraphBridgeRegistry::Instance() {
  static SubgraphBridgeRegistry x;
  return x;
}

void SubgraphBridgeRegistry::Insert(const std::string& op_type,
                                    const TargetType& target,
                                    const cvt_func_type& cvt_func_name) {
  int key = static_cast<int>(target);
  auto it = map_.find(key);
  if (it == map_.end()) {
    map_.insert(std::make_pair(key, std::map<std::string, cvt_func_type>()));
  }
  map_.at(key).insert(std::make_pair(op_type, cvt_func_name));
}

const cvt_func_type& SubgraphBridgeRegistry::Select(
    const std::string& op_type, const TargetType& target) const {
  int key = static_cast<int>(target);
  return map_.at(key).at(op_type);
}

bool SubgraphBridgeRegistry::Exists(const std::string& op_type,
                                    const TargetType& target) const {
  int key = static_cast<int>(target);
  bool found = map_.find(key) != map_.end();
  if (found) {
    found = map_.at(static_cast<int>(key)).find(op_type) != map_.at(key).end();
  }
  return found;
}

}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
