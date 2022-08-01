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

#pragma once
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/backends/x86/fluid/rw_lock.h"
#include "lite/core/variable.h"

namespace paddle {
namespace lite {

class Scope final {
 public:
  Scope()
      : kids_lock_{new lite::fluid::RWLock},
        vars_lock_{new lite::fluid::RWLock},
        rwlock_{new lite::fluid::RWLock} {}
  // delete below two functions to allow pybind to recognise it cannot make a
  // copy
  // link:
  // https://stackoverflow.com/questions/53807248/pybind11-returning-a-pointer-to-a-container-of-unique-ptr
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  ~Scope();

  Scope& NewScope() const;

  Variable* Var(const std::string& name);

  Variable* LocalVar(const std::string& name);

  Variable* FindVar(const std::string& name) const;

  Variable* FindLocalVar(const std::string& name) const;

  void DeleteLocalVar(const std::string& name);

  const Scope* parent() const { return parent_; }
  Scope* MutableParent() { return const_cast<Scope*>(parent_); }

  // Get attribute params stored in parent scopes.
  std::vector<std::string> AttributeVarNames() const;
  // Following the legacy scope interface.
  std::vector<std::string> LocalVarNames() const;

  /// ------------------------------------- helper functions for Tensor
  /// ----------------------------------
  // Create a Tensor variable. This will create a new Variable called `name`.
  Tensor* NewTensor(const std::string& name) {
    auto* var = Var(name);
    return var->GetMutable<Tensor>();
  }

  const Tensor* FindTensor(const std::string& name) {
    auto* var = FindVar(name);
    if (!var) return nullptr;
    return &var->Get<Tensor>();
  }

  Tensor* FindMutableTensor(const std::string& name) {
    auto* var = FindVar(name);
    if (!var) return nullptr;
    return var->GetMutable<Tensor>();
  }

  std::vector<Tensor>* NewTensorList(const std::string& name) {
    auto* var = Var(name);
    return var->GetMutable<std::vector<Tensor>>();
  }

  const std::vector<Tensor>* FindTensorList(const std::string& name) {
    auto* var = FindVar(name);
    if (!var) return nullptr;
    return &var->Get<std::vector<Tensor>>();
  }

  std::vector<Tensor>* FindMutableTensorList(const std::string& name) {
    auto* var = FindVar(name);
    if (!var) return nullptr;
    return var->GetMutable<std::vector<Tensor>>();
  }

 private:
  // Scope in `kids_` are owned by this class.
  mutable std::list<Scope*> kids_;
  const Scope* parent_{nullptr};
  std::map<std::string, std::unique_ptr<Variable>> vars_;
  std::unique_ptr<lite::fluid::RWLock> kids_lock_{nullptr};
  std::unique_ptr<lite::fluid::RWLock> vars_lock_{nullptr};
  std::unique_ptr<lite::fluid::RWLock> rwlock_{nullptr};
};

}  // namespace lite
}  // namespace paddle
