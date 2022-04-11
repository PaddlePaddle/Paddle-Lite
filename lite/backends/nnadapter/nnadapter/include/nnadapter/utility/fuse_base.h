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

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "utility/logging.h"
#include "utility/pattern_matcher.h"

namespace nnadapter {

class FuseBase {
 public:
  using key2nodes_t = std::map<std::string, Node*>;

  virtual ~FuseBase() = default;

  // Returns number of matched subgraphs
  size_t operator()(Graph* graph, core::Model* model) {
    BuildPattern();
    PerformPatternMatcher(graph);

    for (const auto& matched : key2nodes_) {
      InsertNewNode(graph, model, matched);
    }

    DeleteInterNodes(graph);
    return key2nodes_.size();
  }

  // Build a PMPattern using PMNode.
  virtual void BuildPattern() = 0;

  PMNode* OpNode(const std::string& key) {
    return GetOrCreateNode(key)->assert_is_op();
  }

  PMNode* OpNode(const std::string& key, NNAdapterOperationType op_type);

  PMNode* VarNode(const std::string& key);

 protected:
  virtual void InsertNewNode(Graph* graph,
                             core::Model* model,
                             const key2nodes_t& matched) = 0;

  void PerformPatternMatcher(Graph* graph);

  // Delete nodes that are marked as Intermediate
  void DeleteInterNodes(Graph* graph);

  PMNode* GetOrCreateNode(const std::string& key);

 protected:
  PatternMatcher matcher_;
  std::map<std::string, PMNode*> nodes_;
  std::vector<key2nodes_t> key2nodes_;
};

}  // namespace nnadapter
