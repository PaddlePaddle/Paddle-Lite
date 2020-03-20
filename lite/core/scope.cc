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

#include "lite/core/scope.h"

namespace paddle {
namespace lite {

Scope::~Scope() {
  for (auto *x : kids_) {
    if (x) {
      delete x;
    }
  }
}

Scope &Scope::NewScope() const {
  kids_.push_back(new Scope);
  kids_.back()->parent_ = this;
  return *kids_.back();
}

Variable *Scope::Var(const std::string &name) {
  auto *var = FindVar(name);
  if (var) return var;

  // create a new variable.
  vars_.emplace(name, std::unique_ptr<Variable>(new Variable));
  return vars_[name].get();
}

Variable *Scope::FindVar(const std::string &name) const {
  Variable *var{nullptr};
  var = FindLocalVar(name);
  const Scope *cur_scope = this;
  while (!var && cur_scope->parent()) {
    cur_scope = cur_scope->parent();
    var = cur_scope->FindLocalVar(name);
  }

  return var;
}

Variable *Scope::FindLocalVar(const std::string &name) const {
  auto it = vars_.find(name);
  if (it != vars_.end()) {
    return it->second.get();
  }
  return nullptr;
}

// AttributeVarNames will get persistive attribute names stored in parent scope
std::vector<std::string> Scope::AttributeVarNames() const {
  std::vector<std::string> resulted_keys;
  const Scope *cur_scope = this;
  while (cur_scope->parent()) {
    cur_scope = cur_scope->parent();
    auto keys = cur_scope->LocalVarNames();
    resulted_keys.insert(resulted_keys.end(), keys.begin(), keys.end());
  }
  // remove feed and fetch
  std::vector<std::string> skiped_vars = {"feed", "fetch"};
  for (int i = 0; i < skiped_vars.size(); i++) {
    auto iter =
        std::find(resulted_keys.begin(), resulted_keys.end(), skiped_vars[i]);
    while (iter != resulted_keys.end()) {
      resulted_keys.erase(iter);
      iter =
          std::find(resulted_keys.begin(), resulted_keys.end(), skiped_vars[i]);
    }
  }
  return resulted_keys;
}

std::vector<std::string> Scope::LocalVarNames() const {
  std::vector<std::string> keys;
  for (const auto &item : vars_) {
    keys.push_back(item.first);
  }
  return keys;
}

}  // namespace lite
}  // namespace paddle
