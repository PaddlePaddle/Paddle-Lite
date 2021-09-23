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

#include "lite/model_parser/ssa/var_desc.h"

namespace paddle {
namespace lite {
namespace general {
namespace ssa {

std::weak_ptr<VarDesc> VarDesc::latest() {
  std::shared_ptr<VarDesc> desc;
  if (meta_->series.empty()) {
    desc = shared_from_this();
  } else {
    desc = meta_->series.back();
  }
  return desc;
}

std::weak_ptr<VarDesc> VarDesc::Read(const OpDescBase& op_desc) {
  targets_.push_back(&op_desc);
  if (GetType() == VarDataType::LOD_TENSOR_ARRAY) {
    if (mutable_) mutable_ = !mutable_;
  }
  return latest();
}

std::weak_ptr<VarDesc> VarDesc::NewDescendant() {
  meta_->series.emplace_back(std::make_shared<VarDesc>(*this));
  std::weak_ptr<VarDesc> desc = meta_->series.back();
  desc.lock()->SetCount(meta_->series.size());
  desc.lock()->ResetBlockIdx(kInvalidIdx);
  desc.lock()->ClearTargetOps();
  return desc;
}

std::weak_ptr<VarDesc> VarDesc::Written(const OpDescBase& op_desc) {
  std::weak_ptr<VarDesc> desc;
  if (GetType() == VarDataType::LOD_TENSOR ||
      GetType() == VarDataType::LOD_TENSOR_ARRAY) {
    if (mutable_) {
      mutable_ = !mutable_;
      desc = shared_from_this();
    } else if (persistable_) {
      desc = shared_from_this();
    } else {
      desc = NewDescendant();
    }
  } else {
    desc = shared_from_this();
  }
  return desc;
}

std::vector<std::weak_ptr<VarDesc>> VarDesc::series() const {
  std::vector<std::weak_ptr<VarDesc>> series;
  for (auto& kid : meta_->series) {
    series.emplace_back(kid);
  }
  return series;
}

std::string VarDesc::mangled_name() const {
  std::string suffix;
  if (count_) {
    suffix = "__Mangled_" + std::to_string(count_);
  }
  return root_name() + suffix;
}

bool operator<(const VarDesc& x, const VarDesc& y) {
  if (x.root_var_desc() == y.root_var_desc()) {
    return x.count() < y.count();
  }
  return x.mangled_name() < y.mangled_name();
}

bool VarDescLT::operator()(const std::weak_ptr<VarDesc>& lhs,
                           const std::weak_ptr<VarDesc>& rhs) const {
  auto lptr = lhs.lock(), rptr = rhs.lock();
  if (!rptr) return false;
  if (!lptr) return true;
  return *lptr < *rptr;
}

RootVarScope::RootVarScope(const general::BlockDesc& current,
                           RootVarScope* parent) {
  parent_ = parent;
  if (parent) {
    parent->AddKidScope(this);
  }
  for (size_t i = 0; i < current.VarsSize(); ++i) {
    const general::VarDesc* raw_var{current.GetVar<general::VarDesc>(i)};
    AddRootVar(current.Idx(), *raw_var);
    // Add accompanying variables indicating dependencies for the lod tensor
    // array type, both of which are located on the same level of scope.
    if (raw_var->GetType() == VarDescAPI::Type::LOD_TENSOR_ARRAY) {
      const std::string asso_var_name{raw_var->Name() + ".AssociatedVar"};
      general::VarDesc asso_var(asso_var_name);
      asso_var.SetType(VarDescAPI::Type::LOD_TENSOR);
      asso_var.SetPersistable(false);
      AddRootVar(current.Idx(), std::move(asso_var));
    }
  }
}

void RootVarScope::AddRootVar(int32_t block_idx,
                              const general::VarDesc& raw_var) {
  CHECK_EQ(root_vars_.count(raw_var.Name()), 0);
  root_vars_[raw_var.Name()] = std::make_shared<VarDesc>(block_idx, &raw_var);
}

void RootVarScope::AddRootVar(int32_t block_idx, general::VarDesc&& raw_var) {
  CHECK_EQ(root_vars_.count(raw_var.Name()), 0);
  auto var_name = raw_var.Name();
  root_vars_[var_name] =
      std::make_shared<VarDesc>(block_idx, std::move(raw_var));
}

std::vector<std::weak_ptr<VarDesc>> RootVarScope::GetRootVars() const {
  std::vector<std::weak_ptr<VarDesc>> vars;
  for (auto& pair : root_vars_) {
    vars.emplace_back(pair.second);
  }
  return vars;
}

bool RootVarScope::HasRootVarDesc(const std::string& name) const {
  if (root_vars_.find(name) != root_vars_.end()) {
    return true;
  } else if (parent_) {
    return parent_->HasRootVarDesc(name);
  }
  return false;
}

RootVarScope* RootVarScope::GetMutableScopeOfRootVar(const std::string& name) {
  if (root_vars_.find(name) != root_vars_.end()) {
    return this;
  } else if (parent_) {
    return parent_->GetMutableScopeOfRootVar(name);
  }
  return nullptr;
}

std::weak_ptr<VarDesc> RootVarScope::GetRootVarDesc(
    const std::string& name) const {
  if (root_vars_.find(name) != root_vars_.end()) {
    return root_vars_.at(name);
  } else if (parent_) {
    return parent_->GetRootVarDesc(name);
  }
  LOG(FATAL) << "can not find root var in the current block and root block.";
  return {};
}

}  // namespace ssa
}  // namespace general
}  // namespace lite
}  // namespace paddle
