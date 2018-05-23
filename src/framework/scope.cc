/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#include "framework/scope.h"
#include <set>
#include <string>
#include <vector>

namespace paddle_mobile {
namespace framework {

Scope &Scope::NewScope() const {
  std::unique_lock<std::mutex> lock(mutex_);
  kids_.push_back(new Scope(this));
  return *kids_.back();
}

Variable *Scope::Var(const std::string &name) {
  auto *pvar = FindVarLocally(name);
  if (pvar != nullptr) {
    return pvar;
  }

  pvar = new Variable;
  vars_[name] = pvar;
  pvar->name_ = &(vars_.find(name)->first);
  return pvar;
}

//            Variable* Scope::Var(std::string* name) {
//                auto var_name = string::Sprintf("%p.%d", this,
//                vars_.size());
//                if (name != nullptr) {
//                    *name = var_name;
//                }
//                return Var(var_name);
//            }

Variable *Scope::FindVar(const std::string &name) const {
  auto *pvar = FindVarLocally(name);
  if (pvar != nullptr) {
    return pvar;
  }
  return (parent_ == nullptr) ? nullptr : parent_->FindVar(name);
}

const Scope *Scope::FindScope(const Variable *var) const {
  for (auto &name_var : vars_) {
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
  known_vars.reserve(vars_.size());
  for (auto &name_var : vars_) {
    known_vars.emplace_back(name_var.first);
  }
  return known_vars;
}

void Scope::DeleteScope(Scope *scope) const {
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = std::find(kids_.begin(), kids_.end(), scope);
  kids_.erase(it);
  delete scope;
  // deferent
}

void Scope::EraseVars(const std::vector<std::string> &var_names) {
  std::set<std::string> var_set(var_names.begin(), var_names.end());
  for (auto it = vars_.begin(); it != vars_.end();) {
    if (var_set.find(it->first) != var_set.end()) {
      delete it->second;
      it = vars_.erase(it);
    } else {
      ++it;
    }
  }
}

void Scope::Rename(const std::string &origin_name,
                   const std::string &new_name) const {
  auto origin_it = vars_.find(origin_name);
  if (origin_it == vars_.end()) {
    return;
  }
  auto new_it = vars_.find(new_name);
  if (new_it != vars_.end()) {
    return;
  }
  vars_[new_name] = origin_it->second;
  vars_.erase(origin_it);
}
//
//            std::string Scope::Rename(const std::string& origin_name)
//            const {
//                auto var_name = string::Sprintf("%p.%d", this,
//                vars_.size());
//                Rename(origin_name, var_name);
//                return var_name;
//            }

Variable *Scope::FindVarLocally(const std::string &name) const {
  auto it = vars_.find(name);
  if (it != vars_.end()) {
    return it->second;
  }
  return nullptr;
}

}  // namespace framework
}  // namespace paddle_mobile
