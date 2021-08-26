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
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "lite/core/model/general/block_desc.h"
#include "lite/model_parser/ssa/op_desc.h"
#include "lite/model_parser/ssa/var_desc.h"

namespace paddle {
namespace lite {
namespace general {
namespace ssa {

class BlockDesc {
 public:
  static constexpr int32_t kInvalidIdx{-1};

  BlockDesc(const general::BlockDesc& current, BlockDesc* parent);

  explicit BlockDesc(const general::BlockDesc& current)
      : BlockDesc{current, nullptr} {}

  int32_t idx() const { return idx_; }

  const std::list<std::unique_ptr<OpDescBase>>& ops() const { return ops_; }

  const OpDescBase* AddOp(std::unique_ptr<OpDescBase>&& op) {
    ops_.emplace_back(std::move(op));
    return ops_.back().get();
  }

  RootVarScope* mutable_scope() { return scope_.get(); }

  const RootVarScope* scope() const { return scope_.get(); }

  const BlockDesc* parent() const { return parent_; }

  const std::vector<BlockDesc*>& kids() const { return kids_; }

  void AddKid(BlockDesc* desc) { kids_.emplace_back(desc); }

  void SetBlockOpDesc(BlockOpDesc* op) {
    CHECK(op);
    block_op_ = op;
  }

  BlockOpDesc* mutable_block_op() { return block_op_; }

  template <typename InputIt>
  void AddBlockInputs(InputIt first, InputIt last, bool extra = false) {
    if (extra) {
      block_extra_inputs_.insert(first, last);
    } else {
      block_inputs_.insert(first, last);
    }
  }

  template <typename InputIt>
  void AddBlockOutputs(InputIt first, InputIt last) {
    block_outputs_.insert(first, last);
  }

  const std::set<std::weak_ptr<VarDesc>, VarDescLT>& block_inputs() {
    return block_inputs_;
  }

  const std::set<std::weak_ptr<VarDesc>, VarDescLT>& block_outputs() {
    return block_outputs_;
  }

  const std::set<std::weak_ptr<VarDesc>, VarDescLT>& block_extra_inputs() {
    return block_extra_inputs_;
  }

 private:
  std::vector<BlockDesc*> kids_;
  BlockDesc* parent_{nullptr};
  // The root variable scope serves as a symbol table here.
  std::unique_ptr<RootVarScope> scope_;
  int32_t idx_{kInvalidIdx};
  std::list<std::unique_ptr<OpDescBase>> ops_;
  BlockOpDesc* block_op_{nullptr};
  std::set<std::weak_ptr<VarDesc>, VarDescLT> block_inputs_;
  std::set<std::weak_ptr<VarDesc>, VarDescLT> block_outputs_;
  std::set<std::weak_ptr<VarDesc>, VarDescLT> block_extra_inputs_;
};

}  // namespace ssa
}  // namespace general
}  // namespace lite
}  // namespace paddle
