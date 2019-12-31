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

#include "lite/kernels/npu/bridges/registry.h"
#include <utility>

namespace paddle {
namespace lite {
namespace subgraph {

Registry& Registry::Instance() {
  static Registry x;
  return x;
}

void Registry::Insert(const std::string& op_type,
                      const std::string& target,
                      const cvt_func_type& cvt_func_name) {
  auto it = map_.find(target);
  if (it == map_.end()) {
    map_.insert(std::make_pair(
        target, std::unordered_map<std::string, cvt_func_type>()));
  }
  map_.at(target).insert(std::make_pair(op_type, cvt_func_name));
}

const cvt_func_type& Registry::Select(const std::string& op_type,
                                      const std::string& target) const {
  return map_.at(target).at(op_type);
}

bool Registry::Exists(const std::string& op_type,
                      const std::string& target) const {
  bool found = map_.find(target) != map_.end();
  if (found) {
    found = map_.at(target).find(op_type) != map_.at(target).end();
  }
  return found;
}

}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
