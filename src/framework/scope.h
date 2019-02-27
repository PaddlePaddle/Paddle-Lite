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

#pragma once

#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef PADDLE_MOBILE_CL
#include "framework/cl/cl_scope.h"
#endif
#include "framework/variable.h"

namespace paddle_mobile {
namespace framework {

class Scope {
 public:
  Scope() = default;

  ~Scope() {
    for (auto &var : vars_) {
      delete var.second;
    }
    vars_.clear();
    for (auto kid : kids_) {
      delete kid;
    }
    kids_.clear();

#ifdef PADDLE_MOBILE_CL
    delete cl_scope_;
#endif
  }

  Scope &NewScope() const;

  /// Create a variable with given name if it doesn't exist.
  Variable *Var(const std::string &name);

  /// Create a variable with a scope-unique name.
  Variable *Var(std::string *name = nullptr);

  void EraseVars(const std::vector<std::string> &var_names);

  /// Find a variable in the scope or any of its ancestors.  Returns
  /// nullptr if cannot find.
  Variable *FindVar(const std::string &name) const;

  const Scope *parent() const { return parent_; }

  /// Find the scope or an ancestor scope that contains the given
  /// variable.
  const Scope *FindScope(const Variable *var) const;

  void DeleteScope(Scope *scope) const;

  /// Drop all kids scopes belonged to this scope.
  void DropKids();

  // enumerate all the variables current contains.
  std::vector<std::string> LocalVarNames() const;

  // Rename variable to a new name
  void Rename(const std::string &origin_name,
              const std::string &new_name) const;

  // Rename variable to a new name and return the new name
  std::string Rename(const std::string &origin_name) const;

  Variable *FindVarLocally(const std::string &name) const;

#ifdef PADDLE_MOBILE_FPGA
  Variable *Var(const std::string &name, const int id);
  std::vector<Variable *> VarContain(const std::string substring);
  void InsertVar(const std::string str, Variable *var);
  void print_vars();
#endif

#ifdef PADDLE_MOBILE_CL
  CLScope *GetCLScpoe() { return cl_scope_; }
#endif

 private:
  // Call Scope::NewScope for a sub-scope.
  explicit Scope(Scope const *parent) : parent_(parent) {}

  mutable std::unordered_map<std::string, Variable *> vars_;
  mutable std::list<Scope *> kids_;
  Scope const *parent_{nullptr};

#ifdef PADDLE_MOBILE_CL
  CLScope *cl_scope_ = new CLScope();
#endif
};
}  // namespace framework
}  // namespace paddle_mobile
