// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lite/model_parser/general/block_desc.h"
#include "lite/model_parser/ssa/op_desc.h"
#include "lite/model_parser/ssa/scope.h"
#include "lite/model_parser/ssa/var_desc.h"

namespace paddle {
namespace lite {
namespace general {
namespace ssa {

class BlockDesc {
 public:
  BlockDesc(const general::BlockDesc& current, BlockDesc* parent) {
    idx_ = current.Idx();
    if (parent) {
      scope_.reset(new Scope{current, parent->mutable_scope()});
    } else {
      scope_.reset(new Scope{current, nullptr});
    }
  }
  explicit BlockDesc(const general::BlockDesc& current)
      : BlockDesc{current, nullptr} {}

  int32_t idx() const { return idx_; }

  const std::list<OpDesc>& ops() const { return ops_; }

  void AddOp(OpDesc&& op) { ops_.emplace_back(std::forward<OpDesc>(op)); }

  Scope* mutable_scope() { return scope_.get(); }

  const Scope* scope() const { return scope_.get(); }

  const BlockDesc* parent() const { return parent_; }

  const BlockDesc* kid() const { return kid_; }

 private:
  const BlockDesc* kid_{nullptr};
  const BlockDesc* parent_{nullptr};
  std::unique_ptr<Scope> scope_;
  int32_t idx_{-1};
  std::list<OpDesc> ops_;
};

}  // namespace ssa
}  // namespace general
}  // namespace lite
}  // namespace paddle
