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

#include <map>
#include <string>
#include "lite/core/model/base/apis.h"
#include "lite/model_parser/flatbuffers/framework_generated.h"

namespace paddle {
namespace lite {
namespace fbs {

/*
 * The general::OpVersionMap is the internal representation for Ops version.
 * All the internal
 * imprementation should use it, not the pb::OpVersionMap.
 */
class OpVersionMap : public OpVersionMapAPI {
 public:
  OpVersionMap() = default;

  explicit OpVersionMap(proto::OpVersionMap* op_version_map) {
    // op_version_map is not implemented on flatbuffer as
    // it's not useful in inference period.
  }
  std::map<std::string, int32_t> GetOpVersionMap() const override {
    return op_version_map_;
  }
  int32_t GetOpVersionByName(const std::string& name) const override {
    return op_version_map_.at(name);
  }

  void SetOpVersionMap(
      const std::map<std::string, int32_t>& op_version_map) override {
    op_version_map_ = op_version_map;
  }

  void AddOpVersion(const std::string& op_name, int32_t op_version) override {
    op_version_map_[op_name] = op_version;
  }

 private:
  std::map<std::string, int32_t> op_version_map_;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
