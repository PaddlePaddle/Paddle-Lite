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
#include <string>
#include <vector>
#include "lite/core/mir/pattern_matcher_high_api.h"
#include "lite/core/mir/xpu_pattern_matcher.h"

namespace paddle {
namespace lite {
namespace mir {
namespace xpu {

class XPUFuseBase {
 public:
  using key2nodes_t = std::map<std::string, Node*>;

  virtual ~XPUFuseBase() = default;

  // Returns number of matched subgraphs
  size_t operator()(SSAGraph* graph) {
    BuildPattern();
    PerformPatternMatcher(graph);

    for (size_t i = 0; i < key2nodes_.size(); ++i) {
      InsertNewNode(graph, key2nodes_[i], matcher_.extra_input_vars_[i]);
    }

    DeleteInterNodes(graph);
    return key2nodes_.size();
  }

  // Build a PMPattern using PMNode.
  virtual void BuildPattern() = 0;

  // Generate an operator desc with a matched subgraph.
  virtual cpp::OpDesc GenOpDesc(const key2nodes_t& matched) {
    return cpp::OpDesc();
  }

  PMNode* OpNode(const std::string& key) {
    return GetOrCreateNode(key)->assert_is_op();
  }

  PMNode* OpNode(const std::string& key, const std::string& op_type);

  PMNode* VarNode(const std::string& key);

 protected:
  virtual void InsertNewNode(SSAGraph* graph,
                             const key2nodes_t& matched,
                             const std::vector<Node*>& extra_input_vars) = 0;

  void PerformPatternMatcher(SSAGraph* graph);

  // Delete nodes that are marked as Intermediate
  void DeleteInterNodes(SSAGraph* graph);

  PMNode* GetOrCreateNode(const std::string& key);

 protected:
  XPUPatternMatcher matcher_;
  std::map<std::string, PMNode*> nodes_;
  std::vector<key2nodes_t> key2nodes_;
};

}  // namespace xpu
}  // namespace mir
}  // namespace lite
}  // namespace paddle
