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
#include <memory>
#include <string>
#include <vector>
#include "lite/core/model/base/apis.h"
#include "lite/core/model/general/block_desc.h"
#include "lite/core/model/general/op_version_map.h"

namespace paddle {
namespace lite {
namespace general {

/*
 * The general::ProgramDesc is the internal representation for Op. All the
 * internal
 * imprementation should use it, not the pb::ProgramDesc.
 */
class ProgramDesc : public ProgramDescAPI {
 public:
  ProgramDesc() = default;

  ProgramDesc(const ProgramDesc&);

  ProgramDesc(ProgramDesc&&) = default;

  ProgramDesc& operator=(const ProgramDesc&);

  void CopyFrom(const ProgramDesc& other);

  const std::vector<std::unique_ptr<BlockDesc>>& blocks() const {
    return blocks_;
  }

  size_t BlocksSize() const override { return blocks_.size(); }

  void ClearBlocks() override { blocks_.clear(); }

  template <typename T>
  T* GetBlock(int32_t idx);

  template <typename T>
  T const* GetBlock(int32_t idx) const;

  std::vector<std::unique_ptr<BlockDesc>>& GetBlocks() { return blocks_; }

  template <typename T>
  T* AddBlock();

  /////////////////////////////////////////////////////////////////
  // Name: OpVersionMap
  // Description: a map that strores paddle ops version
  /////////////////////////////////////////////////////////////////
  bool HasOpVersionMap() const override {
    return !(op_version_map_.GetOpVersionMap().empty());
  }

  template <typename T>
  T* GetOpVersionMap();

  void SetOpVersionMap(std::map<std::string, int32_t> op_version_map);

  // Just return default versoin
  // TODO(sangoly): refine this
  bool HasVersion() const override { return true; }

  int64_t Version() const override { return version_; }

  void SetVersion(int64_t version) override { version_ = version; }

 private:
  int64_t version_;
  OpVersionMap op_version_map_;
  std::vector<std::unique_ptr<BlockDesc>> blocks_;
};

}  // namespace general
}  // namespace lite
}  // namespace paddle
