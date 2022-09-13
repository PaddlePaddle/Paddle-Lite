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

/*
 * This file implements the pattern matcher for op fusion, which is from
 * PaddleLite
 */

#pragma once

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "core/types.h"
#include "utility/logging.h"

namespace nnadapter {

class PatternMatcher {
 public:
  // This is a simple representation of a model. It holds the pointer of the
  // operand and operation to avoid changing the model during the op fusion.
  struct Node {
    explicit Node(core::Operand* o) : operand(o) {}
    explicit Node(core::Operation* o) : operation(o) {}
    bool IsOperand() const;
    bool IsOperation() const;
    core::Operand* operand{nullptr};
    core::Operation* operation{nullptr};
    std::vector<Node*> inlinks{};
    std::vector<Node*> outlinks{};
  };
  using Condition = std::function<bool(const Node*)>;
  /*
   * A pattern in a graph, which defined with pattern nodes and edges. Most
   * graph patterns can be divided into the nodes and the links between them.
   *
   * For example, the FullyConnected fusion need to filter the NNADAPTER_MATMUL
   * and NNADAPTER_ADD operations from the model, the NNADAPTER_MATMUL's output
   * should have only one consumer which is the NNADAPTER_ADD. This pattern can
   * be
   * defined as with the following pseudo codes:
   *   // Create two operation patterns
   *   matmul = CreatePattern("matmul", NNADAPTER_MATMUL);
   *   add = CreatePattern("add", NNADAPTER_ADD);
   *   // Create the operand patterns
   *   output = CreatePattern("output") \
   *              ->IsOperationOutputOperand(NNADAPTER_MATMUL, 0) \
   *              ->IsOperationInputOperand(NNADAPTER_ADD, 0) \
   *              ->IsIntermediate();
   *   // Create the topological connections for the patterns
   *   matmul >> output;
   *   output >> add;
   *
   * One can add more specific asserts for the pattern nodes or edges, both the
   * operation and operand pattern nodes can be ruled in MatchCondition(...).
   *
  */
  struct Pattern;
  using Edge = std::pair<Pattern*, Pattern*>;
  using Subgraph = std::map<Pattern*, Node*>;
  struct Pattern {
    explicit Pattern(std::vector<Edge>* e, NNAdapterOperationType t);
    // Link self to the other pattern
    Pattern& operator>>(Pattern& other);
    // Link self to other patterns
    Pattern& operator>>(std::vector<Pattern*>& others);
    // Link other patterns to self
    friend Pattern& operator>>(std::vector<Pattern*>& others, Pattern& self);
    bool MatchAllConditions(const Node* node) const;
    // Utility conditions
    Pattern* IsOperand();
    Pattern* IsOperation(NNAdapterOperationType type = NNADAPTER_UNKNOWN);
    Pattern* IsConstantOperand();
    Pattern* IsVariableOperand();
    Pattern* IsConstantCopyOperand();
    Pattern* IsConstantReferenceOperand();
    Pattern* IsTemporaryVariableOperand();
    Pattern* IsTemporaryShapeOperand();
    Pattern* IsOperationInputOperand(NNAdapterOperationType type,
                                     int index = -1);
    Pattern* IsOperationOutputOperand(NNAdapterOperationType type,
                                      int index = -1);
    Pattern* CheckInputCount(int num);
    Pattern* CheckOutputCount(int num);
    // Mark the pattern matched node to be deleted, so its inlinks and outlinks
    // should be inside a matched subgraph.
    Pattern* IsIntermediate();
    // Add a custom condition
    Pattern* MatchCondition(const Condition& condition);
    Pattern& operator=(const Pattern&) = delete;
    Pattern(const Pattern&) = delete;
    Pattern(Pattern&& other) = default;
    std::vector<Edge>* edges;
    NNAdapterOperationType type{NNADAPTER_UNKNOWN};
    bool intermediate{false};
    std::vector<Condition> conditions;
  };

  virtual ~PatternMatcher() = default;

  // Return the number of matched subgraphs
  size_t Apply(core::Model* model);
  virtual void BuildPattern() = 0;
  // Create or a operand or operation pattern with a unique key
  Pattern* CreatePattern(const std::string& key,
                         NNAdapterOperationType type = NNADAPTER_UNKNOWN);

 protected:
  // Handling the matched subgraphs, such as inserting a new operation and some
  // operands, then delete the intermediate operand and operations if returns
  // true, otherwise ignore it.
  virtual bool HandleMatchedResults(
      core::Model* model, const std::map<std::string, Node*>& nodes) = 0;
  // Convert the operands and operations of the core::Model to the nodes with
  // inlinks and outlinks
  void BuildNodes(core::Model* model, std::list<Node>* nodes);
  bool MarkPatterns(std::list<Node>* nodes);
  // Detect all the pattern and output the hit records.
  std::vector<Subgraph> DetectPatterns();
  void UniquePatterns(std::vector<PatternMatcher::Subgraph>* subgraphs);
  void ValidatePatterns(std::vector<PatternMatcher::Subgraph>* subgraphs);
  void RemoveOverlappedPatterns(std::vector<Subgraph>* subgraphs);

 protected:
  std::map<std::string, std::unique_ptr<Pattern>> patterns_;
  std::vector<Edge> edges_;
  std::map<const Pattern*, std::set<Node*>> pattern2nodes_;
};

}  // namespace nnadapter
