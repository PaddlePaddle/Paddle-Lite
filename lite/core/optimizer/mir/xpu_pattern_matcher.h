// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <utility>
#include <vector>
#include "lite/core/mir/pattern_matcher.h"

namespace paddle {
namespace lite {
namespace mir {
namespace xpu {

/*
 * PatternMatcher helps to detect the specific patterns in the graph.
 * Input a pattern, output a list of the matched subgraphs/nodes.
 * This helper can be used to support fuse(conv+batchnorm => batchnorm e.g.).
 *
 * The algorithm has three phases:
 *   1. Mark the nodes that match the defined PMNodes in a PMPattern,
 *   2. Extend a PMNode to subgraphs by deducing the connection relation defined
 *      in PAPattern(the edges),
 *   3. Get the filtered subgraphs and treat them with a pre-defined handler.
 *
 * Usage:
 *    // Create a matcher
 *    PatternMatcher matcher;
 *    // Define the matcher's pattern, by adding PMNode and define the edges.
 *    auto* node0 = matcher.mutable_pattern().AddNode(...)
 *    auto* node1 = matcher.mutable_pattern().AddNode(...)
 *    node0->teller = some lambda.
 *    node1->teller = some lambda.
 *    matcher.mutable_pattern().AddEdge(node0, node1);
 *    // Create an handler, to define the behavior of treating the filtered
 *    // subgraphs that comply with the patterns.
 *    PatternMatcher::handle_t handler = some labmda
 *    // Execute the matcher.
 *    matcher(&graph, handler);
 */
struct XPUPatternMatcher {
  using subgraph_t = std::map<PMNode*, Node*>;

  // Operate on the detected pattern.
  using handle_t =
      std::function<void(const subgraph_t& /*hitted pattern*/, SSAGraph*)>;

  void operator()(SSAGraph* graph, handle_t handler);

  const PMPattern& pattern() const { return pattern_; }
  PMPattern* mutable_pattern() { return &pattern_; }

  // Mark the nodes that fits the pattern.
  bool MarkPMNodesInGraph(SSAGraph* graph);

  // Detect all the pattern and output the hit records.
  std::vector<subgraph_t> DetectPatterns();

  // Remove duplicate patterns.
  void UniquePatterns(std::vector<subgraph_t>* subgraphs);

  // Remove overlapped match subgraphs, when overlapped, keep the previous one.
  // The intermediate PMNodes will be removed, so can't shared by multiple
  // patterns.
  void RemoveOverlappedMatch(std::vector<subgraph_t>* subgraphs);

  // Validate whether the intermediate nodes are linked by external nodes.
  void ValidateByNodeRole(std::vector<subgraph_t>* subgraphs);

  using hit_rcd_t =
      std::pair<Node* /*node in graph*/, PMNode* /*node in pattern*/>;
  PMPattern pattern_;
  std::map<const PMNode*, std::set<Node*>> pmnodes2nodes_;
  std::vector<std::vector<Node*>> extra_input_vars_;
};

}  // namespace xpu
}  // namespace mir
}  // namespace lite
}  // namespace paddle
