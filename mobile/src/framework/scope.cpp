/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "framework/scope.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace paddle_mobile {
namespace framework {

Scope &Scope::NewScope() const {
  kids_.push_back(new Scope(this));
  return *kids_.back();
}

Variable *Scope::Var() {
  auto *pvar = new Variable;
  unnamed_vars_.push_back(pvar);
  return pvar;
}

Variable *Scope::Var(const std::string &name) {
  auto *pvar = FindVarLocally(name);
  if (pvar != nullptr) {
    return pvar;
  }
  pvar = new Variable;
  named_vars_[name] = pvar;
  pvar->name_ = named_vars_.find(name)->first;
  return pvar;
}

Variable *Scope::FindVar(const std::string &name) const {
  auto *pvar = FindVarLocally(name);
  if (pvar != nullptr) {
    return pvar;
  }
  return (parent_ == nullptr) ? nullptr : parent_->FindVar(name);
}

const Scope *Scope::FindScope(const Variable *var) const {
  for (auto &name_var : named_vars_) {
    if (name_var.second == var) {
      return this;
    }
  }
  return (parent_ == nullptr) ? nullptr : parent_->FindScope(var);
}

void Scope::DropKids() {
  for (Scope *s : kids_) {
    delete s;
  }
  kids_.clear();
}

std::vector<std::string> Scope::LocalVarNames() const {
  std::vector<std::string> known_vars;
  known_vars.reserve(named_vars_.size());
  for (auto &name_var : named_vars_) {
    known_vars.emplace_back(name_var.first);
  }
  return known_vars;
}

void Scope::DeleteScope(Scope *scope) const {
  auto it = std::find(kids_.begin(), kids_.end(), scope);
  kids_.erase(it);
  delete scope;
}

void Scope::EraseVars(const std::vector<std::string> &var_names) {
  std::set<std::string> var_set(var_names.begin(), var_names.end());
  for (auto it = named_vars_.begin(); it != named_vars_.end();) {
    if (var_set.find(it->first) != var_set.end()) {
      delete it->second;
      it = named_vars_.erase(it);
    } else {
      ++it;
    }
  }
}

void Scope::Rename(const std::string &origin_name,
                   const std::string &new_name) const {
  auto origin_it = named_vars_.find(origin_name);
  if (origin_it == named_vars_.end()) {
    return;
  }
  auto new_it = named_vars_.find(new_name);
  if (new_it != named_vars_.end()) {
    return;
  }
  named_vars_[new_name] = origin_it->second;
  named_vars_.erase(origin_it);
}

Variable *Scope::FindVarLocally(const std::string &name) const {
  auto it = named_vars_.find(name);
  if (it != named_vars_.end()) {
    return it->second;
  }
  return nullptr;
}

#ifdef PADDLE_MOBILE_FPGA
Variable *Scope::Var(const std::string &name, const int id) {
  return Var(name + std::to_string(id));
}

std::vector<Variable *> Scope::VarContain(const std::string substring,
                                          int *min) {
  std::vector<Variable *> v;

  int temp = 9999;
  auto len0 = substring.length();
  for (auto pair : named_vars_) {
    if (pair.first.find(substring) == 0) {
      v.push_back(pair.second);
      auto len1 = pair.first.length();
      int index = std::stoi(pair.first.substr(len0, len1));
      if (index < temp) {
        temp = index;
      }
    }
  }
  *min = temp;
  return v;
}

void Scope::print_vars() {
  DLOG << "====================start to print variables=================";
  for (auto pair : named_vars_) {
    DLOG << pair.first;
  }
  DLOG << "==================complete printing variables================";
}
#endif

}  // namespace framework
}  // namespace paddle_mobile
