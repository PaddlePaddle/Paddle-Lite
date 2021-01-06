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
#include <vector>
#include "lite/backends/opencl/utils/cache_generated.h"

namespace paddle {
namespace lite {
namespace fbs {
namespace opencl {

class Cache {
 public:
  explicit Cache(
      const std::map<std::string, std::vector<std::vector<uint8_t>>>& map)
      : binary_map_{map} {}
  explicit Cache(const std::vector<uint8_t>& buffer);
  void CopyDataToBuffer(std::vector<uint8_t>* buffer) const;
  const std::map<std::string, std::vector<std::vector<uint8_t>>>& GetBinaryMap()
      const {
    return binary_map_;
  }

 private:
  void SyncFromFbs(const paddle::lite::fbs::opencl::proto::Cache* desc);
  flatbuffers::DetachedBuffer SyncToFbs() const;
  std::map<std::string, std::vector<std::vector<uint8_t>>> binary_map_;
};

}  // namespace opencl
}  // namespace fbs
}  // namespace lite
}  // namespace paddle
