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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "lite/model_parser/general/block_desc.h"
#include "lite/model_parser/general/var_desc.h"

namespace paddle {
namespace lite {
namespace general {
namespace ssa {

class OpDescBase;

// Strengthen the original var desc to perform static single assignment
// processing.

class VarDesc : public std::enable_shared_from_this<VarDesc> {
 public:
  static constexpr int32_t kInvalidIdx{-1};

  VarDesc() = delete;

  VarDesc(const VarDesc&) = default;

  VarDesc(int32_t block_idx, const general::VarDesc* raw_desc)
      : block_idx_{block_idx}, meta_{std::make_shared<MetaInfo>()} {
    meta_->raw_desc = raw_desc;
  }

  int32_t block_idx() const { return block_idx_; }

  void ResetBlockIdx(int32_t idx) { block_idx_ = idx; }

  const general::VarDesc* root_var_desc() const { return meta_->raw_desc; }

  std::string root_name() const { return meta_->raw_desc->Name(); }

  size_t count() const { return count_; }

  std::string mangled_name() const;

  VarDataType GetType() const { return meta_->raw_desc->GetType(); }

  bool Persistable() const { return meta_->raw_desc->Persistable(); }

  std::weak_ptr<VarDesc> Read(const OpDescBase& op_desc);

  std::weak_ptr<VarDesc> Written(const OpDescBase& op_desc);

  std::vector<std::weak_ptr<VarDesc>> series() const;

  const std::vector<OpDescBase const*>& target_ops() const { return targets_; }

  std::weak_ptr<VarDesc> latest();

 private:
  std::weak_ptr<VarDesc> NewDescendant();

  void SetCount(size_t count) { count_ = count; }

  void ClearTargetOps() { targets_.clear(); }

 private:
  struct MetaInfo {
    general::VarDesc const* raw_desc{nullptr};
    std::vector<std::shared_ptr<VarDesc>> series;
  };
  int32_t block_idx_{kInvalidIdx};
  std::shared_ptr<MetaInfo> meta_;
  bool mutable_{true};
  std::vector<OpDescBase const*> targets_;
  size_t count_{0};
};

bool operator<(const VarDesc& x, const VarDesc& y);

// In order to avoid the randomness brought by the raw pointer value
// as a key, an ordered function is used here.

struct VarDescLT {
  bool operator()(const std::weak_ptr<VarDesc>& lhs,
                  const std::weak_ptr<VarDesc>& rhs) const;
};

class RootVarScope {
 public:
  RootVarScope(const general::BlockDesc& current, RootVarScope* parent);

  explicit RootVarScope(const general::BlockDesc& current)
      : RootVarScope{current, nullptr} {}

  std::vector<std::weak_ptr<VarDesc>> GetRootVars() const;

  std::weak_ptr<VarDesc> GetRootVarDesc(const std::string& name) const;

  const RootVarScope* parent() const { return parent_; }

  const RootVarScope* kid() const { return kid_; }

 protected:
  void SetKidScope(const RootVarScope& kid) { kid_ = &kid; }

 private:
  const RootVarScope* kid_{nullptr};
  const RootVarScope* parent_{nullptr};
  std::map<std::string, std::shared_ptr<VarDesc>> root_vars_;
};

}  // namespace ssa
}  // namespace general
}  // namespace lite
}  // namespace paddle
