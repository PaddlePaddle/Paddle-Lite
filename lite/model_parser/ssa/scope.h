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

namespace paddle {
namespace lite {
namespace general {
namespace ssa {

class VarDesc;

class Scope {
 public:
  Scope(const general::BlockDesc& current, Scope* parent) {
    parent_ = parent;
    if (parent) {
      parent->SetKidScope(*this);
    }
    for (size_t i = 0; i < current.VarsSize(); ++i) {
      const auto* raw_var{current.GetVar<general::VarDesc>(i)};
      root_vars_[raw_var->Name()] =
          std::make_shared<VarDesc>(current.Idx(), raw_var);
    }
  }

  explicit Scope(const general::BlockDesc& current) : Scope{current, nullptr} {}

  std::vector<std::weak_ptr<VarDesc>> GetRootVars() const {
    std::vector<std::weak_ptr<VarDesc>> vars;
    for (auto& pair : root_vars_) {
      vars.emplace_back(pair.second);
    }
    return vars;
  }

  std::weak_ptr<VarDesc> GetRootVarDesc(const std::string& name) const {
    if (root_vars_.find(name) != root_vars_.end()) {
      return root_vars_.at(name);
    } else if (parent_) {
      return parent_->GetRootVarDesc(name);
    } else {
      LOG(FATAL)
          << "can not find root var in the current block and root block.";
      return {};
    }
  }

  const Scope* parent() const { return parent_; }

  const Scope* kid() const { return kid_; }

 protected:
  void SetKidScope(const Scope& kid) { kid_ = &kid; }

 private:
  const Scope* kid_{nullptr};
  const Scope* parent_{nullptr};
  std::map<std::string, std::shared_ptr<VarDesc>> root_vars_;
};

}  // namespace ssa
}  // namespace general
}  // namespace lite
}  // namespace paddle
