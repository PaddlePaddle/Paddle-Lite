// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <utility>
#include <vector>
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

class Node {
 public:
  std::list<Node*> inlinks;
  std::list<Node*> outlinks;

  enum class Role {
    kOperation = 0,
    kOperand = 1,
    kUnk = 2,
  };

  Node() = default;

  core::Operation* operation() const {
    NNADAPTER_CHECK(IsOperation());
    return operation_;
  }

  core::Operand* operand() const {
    NNADAPTER_CHECK(IsOperand());
    return operand_;
  }

  friend std::ostream& operator<<(std::ostream& os, Node& other) {
    os << static_cast<int>(other.role_) << " ";
    if (!other.IsRoleSet()) {
      os << "Unk role node";
    }
    if (other.IsOperation()) {
      auto& operation = other.AsOperation();
      os << "Operation " << OperationTypeToString(operation.type);
    }
    if (other.IsOperand()) {
      auto& operand = other.AsOperand();
      os << "Operand " << OperandToString(&operand);
    }
    return os;
  }

  core::Operand& AsOperand(core::Operand& var) {  // NOLINT
    role_ = Role::kOperand;
    operand_ = &var;
    return *operand_;
  }

  core::Operation& AsOperation(core::Operation& op) {  // NOLINT
    role_ = Role::kOperation;
    operation_ = &op;
    return *operation_;
  }

  // Set roles.
  core::Operand& AsOperand() {
    if (role_ != Role::kUnk) {
      NNADAPTER_CHECK(role_ == Role::kOperand);
      return *operand_;
    }
    role_ = Role::kOperand;
    return *operand_;
  }

  core::Operation& AsOperation() {
    if (role_ != Role::kUnk) {
      NNADAPTER_CHECK(role_ == Role::kOperation);
      return *operation_;
    }
    role_ = Role::kOperation;
    return *operation_;
  }

  // Check roles.
  bool IsRoleSet() const { return role_ != Role::kUnk; }
  bool IsOperation() const { return role_ == Role::kOperation; }
  bool IsOperand() const { return role_ == Role::kOperand; }

  // remap to core::model
  core::Model NodeToModelGraph();

 private:
  // Either operation_ or operand_ is used.
  core::Operation* operation_;
  core::Operand* operand_;
  Role role_{Role::kUnk};
};

}  // namespace nnadapter
