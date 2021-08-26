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
#include <utility>
#include <vector>

#include "lite/core/model/general/block_desc.h"
#include "lite/core/model/general/var_desc.h"

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
    SetPersistable(*raw_desc);
    meta_->raw_desc =
        std::unique_ptr<const general::VarDesc,
                        std::function<void(const general::VarDesc*)>>(
            raw_desc, [](const general::VarDesc*) {});
  }

  VarDesc(int32_t block_idx, general::VarDesc&& raw_desc)
      : block_idx_{block_idx}, meta_{std::make_shared<MetaInfo>()} {
    SetPersistable(raw_desc);
    meta_->raw_desc =
        std::unique_ptr<const general::VarDesc,
                        std::function<void(const general::VarDesc*)>>(
            new general::VarDesc(std::move(raw_desc)),
            [](const general::VarDesc* p) { delete p; });
  }

  int32_t block_idx() const { return block_idx_; }

  void ResetBlockIdx(int32_t idx) { block_idx_ = idx; }

  const general::VarDesc* root_var_desc() const {
    return meta_->raw_desc.get();
  }

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

  void SetPersistable(const general::VarDesc& raw_desc) {
    if (raw_desc.Persistable()) {
      persistable_ = true;
      mutable_ = false;
    }
  }

 private:
  struct MetaInfo {
    std::unique_ptr<const general::VarDesc,
                    std::function<void(const general::VarDesc*)>>
        raw_desc;
    std::vector<std::shared_ptr<VarDesc>> series;
  };
  int32_t block_idx_{kInvalidIdx};
  std::shared_ptr<MetaInfo> meta_;
  bool mutable_{true};
  bool persistable_{false};
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

  void AddRootVar(int32_t block_idx, const general::VarDesc& raw_var);

  void AddRootVar(int32_t block_idx, general::VarDesc&& raw_var);

  std::vector<std::weak_ptr<VarDesc>> GetRootVars() const;

  bool HasRootVarDesc(const std::string& name) const;

  std::weak_ptr<VarDesc> GetRootVarDesc(const std::string& name) const;

  const RootVarScope* parent() const { return parent_; }

  const std::vector<RootVarScope*>& kids() const { return kids_; }

 protected:
  void AddKidScope(RootVarScope* kid) {
    CHECK(kid);
    kids_.emplace_back(kid);
  }

  RootVarScope* GetMutableScopeOfRootVar(const std::string& name);

  const std::map<std::string, std::shared_ptr<VarDesc>>& GetRootVarsMap() {
    return root_vars_;
  }

 private:
  std::vector<RootVarScope*> kids_;
  RootVarScope* parent_{nullptr};
  std::map<std::string, std::shared_ptr<VarDesc>> root_vars_;
};

}  // namespace ssa
}  // namespace general
}  // namespace lite
}  // namespace paddle
