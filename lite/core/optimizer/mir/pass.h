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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "lite/core/optimizer/mir/node.h"
#include "lite/core/optimizer/mir/ssa_graph.h"
#include "lite/utils/varient.h"

namespace paddle {
namespace lite {
namespace mir {

class Pass {
 public:
  // Some appoint here, one pass should be only one of the following kinds.
  enum class Kind {
    // Will modify the program/graph topology.
    kProgramWise = 0,
    // Will modify the statement, with the graph topology fixed.
    kStmtWise,
    // Will not modify the IR, just collect information or visualization.
    kDebug,
  };

  explicit Pass(Kind kind) : kind_(kind) {}

  virtual void Apply(const std::unique_ptr<SSAGraph>& graph) = 0;

  void set_name(const std::string& name) { name_ = name; }
  const std::string& name() const { return name_; }

  void set_doc(const std::string& doc) { doc_ = doc; }
  const std::string& doc() const { return doc_; }

  // Some passes only apply to qualified targets, which need to be explicitly
  // declared.

  // Bind targets. At runtime, there must be one device in the bound targets.
  void BindTargets(const std::set<TargetType>& targets) {
    for (const auto& target : targets) {
      const std::set<TargetType>& universe = ExpandValidTargets(target);
      std::set_union(bound_targets_.begin(),
                     bound_targets_.end(),
                     universe.begin(),
                     universe.end(),
                     std::inserter(bound_targets_, bound_targets_.begin()));
    }
  }

  // Exclude targets. At runtime, there must be one device in the bound targets.
  // Disable the pass if one of the valid devices is in the excluded targets.
  void ExcludeTargets(const std::set<TargetType>& targets) {
    for (const auto& target : targets) {
      const std::set<TargetType>& universe = ExpandValidTargets(target);
      std::set<TargetType> updated_bound_targets;
      std::set_difference(
          bound_targets_.begin(),
          bound_targets_.end(),
          universe.begin(),
          universe.end(),
          std::inserter(updated_bound_targets, updated_bound_targets.begin()));
      bound_targets_ = updated_bound_targets;
      std::set_union(
          excluded_targets_.begin(),
          excluded_targets_.end(),
          universe.begin(),
          universe.end(),
          std::inserter(excluded_targets_, excluded_targets_.begin()));
    }
  }

  // Get all bound targets.
  const std::set<TargetType>& BoundTargets() const { return bound_targets_; }
  // Get all excluded targets.
  const std::set<TargetType>& ExcludedTargets() const {
    return excluded_targets_;
  }

  // Some passes are only available on qualified kernels and need to be
  // explicitly declared.
  // Bind kernels. All kernels bound at runtime must be registered.
  void BindKernels(
      const std::map<std::string, std::set<lite_api::Place>>& kernels) {
    bound_kernels_ = kernels;
  }
  // Get all bound kernels.
  const std::map<std::string, std::set<lite_api::Place>>& GetBoundKernels()
      const {
    return bound_kernels_;
  }
  // Add one kernel to the bound kernels.
  void BindKernel(const std::string& kernel_name,
                  const lite_api::Place& place) {
    if (!bound_kernels_.count(kernel_name)) {
      bound_kernels_.insert({kernel_name, {place}});
    } else {
      bound_kernels_.at(kernel_name).insert(place);
    }
  }

  Kind kind() const { return kind_; }
  bool is_debug_pass() const { return kind_ == Kind::kDebug; }
  bool is_program_pass() const { return kind_ == Kind::kProgramWise; }
  bool is_stmt_pass() const { return kind_ == Kind::kStmtWise; }

  virtual ~Pass() = default;

  bool HasAttr(const std::string& attr_name) const {
    return pass_attrs_.count(attr_name) > 0;
  }

  // Set a pointer to the attribute. Specific pass itself takes ownership of the
  // attribute.
  template <typename AttrType>
  void SetAttr(const std::string& attr_name, const AttrType* attr) {
    VLOG(4) << "Setting the attribute " << attr_name << " for the pass "
            << name_;
    pass_attrs_[attr_name].set<const AttrType>(*attr);
  }

  // Get a reference to the attribute previously set.
  template <typename AttrType>
  const AttrType& GetAttr(const std::string& attr_name) const {
    CHECK(pass_attrs_.count(attr_name))
        << attr_name << " attr not register for pass " << name_;
    return pass_attrs_.at(attr_name).get<const AttrType>();
  }

 private:
  const Kind kind_;
  std::string name_;
  std::string doc_;
  std::set<TargetType> bound_targets_;
  std::set<TargetType> excluded_targets_;
  std::map<std::string, std::set<lite_api::Place>> bound_kernels_;
  std::map<std::string, variant<Node, std::vector<Node*>>> pass_attrs_;
};

// Different kinds.
class ProgramPass : public Pass {
 public:
  ProgramPass() : Pass(Kind::kProgramWise) {}
};

class StmtPass : public Pass {
 public:
  StmtPass() : Pass(Kind::kStmtWise) {}
};

class DebugPass : public Pass {
 public:
  DebugPass() : Pass(Kind::kDebug) {}
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
