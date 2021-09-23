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

#include <map>
#include <string>
#include <vector>
#include "lite/core/framework.pb.h"
#include "lite/core/model/base/apis.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace pb {

class OpVersionMap : public OpVersionMapAPI {
 public:
  OpVersionMap() = delete;

  explicit OpVersionMap(framework::proto::OpVersionMap* op_version_map)
      : op_version_map_(op_version_map) {
    CHECK(op_version_map_);
  }

  framework::proto::OpVersionMap* Proto() { return op_version_map_; }

  const framework::proto::OpVersionMap& ReadonlyProto() const {
    return *op_version_map_;
  }

  std::map<std::string, int32_t> GetOpVersionMap() const override;
  int32_t GetOpVersionByName(const std::string& name) const override;
  void SetOpVersionMap(
      const std::map<std::string, int32_t>& op_version_map) override;
  void AddOpVersion(const std::string& op_name, int32_t op_version) override;

 private:
  framework::proto::OpVersionMap* op_version_map_;  // not_own
};

}  // namespace pb
}  // namespace lite
}  // namespace paddle
