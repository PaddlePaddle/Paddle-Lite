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

#include "lite/model_parser/pb/op_version_map.h"

namespace paddle {
namespace lite {
namespace pb {

std::map<std::string, int32_t> OpVersionMap::GetOpVersionMap() const {
  std::map<std::string, int32_t> op_version_map;
  for (int i = 0; i < op_version_map_->pair_size(); i++) {
    auto& op_version_pair = op_version_map_->pair(i);
    const std::string& op_name = op_version_pair.op_name();
    int32_t op_version_id = op_version_pair.op_version().version();
    op_version_map[op_name] = op_version_id;
  }
  return op_version_map;
}

int32_t OpVersionMap::GetOpVersionByName(const std::string& name) const {
  for (int i = 0; i < op_version_map_->pair_size(); i++) {
    auto& op_version_pair = op_version_map_->pair(i);
    const std::string& op_name = op_version_pair.op_name();
    if (op_name == name) {
      int32_t op_version_id = op_version_pair.op_version().version();
      return op_version_id;
    }
  }
  // Not found: return -1 as default value.
  return -1;
}

void OpVersionMap::SetOpVersionMap(
    const std::map<std::string, int32_t>& op_version_map) {
  op_version_map_->Clear();
  for (auto iter = op_version_map.begin(); iter != op_version_map.end();
       iter++) {
    AddOpVersion(iter->first, iter->second);
  }
}

void OpVersionMap::AddOpVersion(const std::string& op_name,
                                int32_t op_version) {
  // Create a new op_version_pair
  auto* new_op_version_pair = op_version_map_->add_pair();
  // 1. Set name of this op_version_pair
  new_op_version_pair->set_op_name(op_name);
  // 2.1 Get op_version
  auto* op_version_proto = new_op_version_pair->mutable_op_version();
  // 2.2 Set the value of op_version
  op_version_proto->set_version(op_version);
}

}  // namespace pb
}  // namespace lite
}  // namespace paddle
