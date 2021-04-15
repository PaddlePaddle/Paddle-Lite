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

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "lite/core/mir/dot.h"
#include "lite/core/mir/xpu_pattern_matcher.h"
#include "lite/core/op_lite.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace mir {
namespace xpu {

void XPUPatternMatcher::operator()(SSAGraph *graph,
                                   XPUPatternMatcher::handle_t handler) {
  if (!MarkPMNodesInGraph(graph)) {
    return;
  }

  auto subgraphs = DetectPatterns();
  UniquePatterns(&subgraphs);
  RemoveOverlappedMatch(&subgraphs);
  ValidateByNodeRole(&subgraphs);

  if (subgraphs.empty()) return;
  LOG(INFO) << "detected " << subgraphs.size() << " subgraph";
  int id = 0;
  for (auto &g : subgraphs) {
    VLOG(3) << "optimizing #" << id++ << " subgraph";
    handler(g, graph);
  }
}

bool XPUPatternMatcher::MarkPMNodesInGraph(SSAGraph *graph) {
  VLOG(3) << "mark pmnodes in graph";
  if (graph->nodes().empty()) return false;
  for (auto &node : graph->mutable_nodes()) {
    for (const auto &pmnode : pattern_.nodes()) {
      if (pmnode->Tell(&node)) {
        pmnodes2nodes_[pmnode.get()].insert(&node);
      }
    }
  }
  // Check to early stop if some PMNode can't find matched Node.
  for (auto &pmnode : pattern_.nodes()) {
    if (!pmnodes2nodes_.count(pmnode.get())) {
      VLOG(4) << pmnode->name() << " can't find matched Node, early stop";
      // return false;
    }
  }
  VLOG(3) << pmnodes2nodes_.size() << " nodes marked";

  return !pmnodes2nodes_.empty();
}

// The intermediate Nodes can only link to the nodes inside the pattern, or this
// subgraph will be droped.
void XPUPatternMatcher::ValidateByNodeRole(
    std::vector<PatternMatcher::subgraph_t> *subgraphs) {
  subgraphs->erase(
      std::remove_if(subgraphs->begin(),
                     subgraphs->end(),
                     [](const XPUPatternMatcher::subgraph_t &subgraph) -> bool {
                       // Collect the inlinks and outlinks.
                       std::set<Node *> ios;
                       for (auto &item : subgraph) {
                         ios.insert(item.second);
                       }
                       for (auto &item : subgraph) {
                         if (item.first->IsIntermediate()) {
                           for (auto *x : item.second->outlinks) {
                             if (!ios.count(x)) {
                               return true;
                             }
                           }
                         }
                       }
                       return false;
                     }),
      subgraphs->end());

  for (auto &subgraph : *subgraphs) {
    std::set<Node *> ios;
    for (auto &item : subgraph) {
      ios.insert(item.second);
    }
    extra_input_vars_.emplace_back();
    for (auto &item : subgraph) {
      for (auto *x : item.second->inlinks) {
        if (x->IsArg() && ios.count(x) == 0) {
          // extra weight var
          extra_input_vars_.back().push_back(x);
        }
      }
    }
  }
}

struct HitGroup {
  std::map<PMNode *, Node *> roles;

  bool Match(Node *node, PMNode *pat) {
    if (nodes_.count(node)) {
      if (roles.count(pat) && roles[pat] == node) return true;
      return false;
    } else {
      if (roles.count(pat) && roles[pat] != node) return false;
      return true;
    }
  }

  void Register(Node *node, PMNode *pat) {
    roles[pat] = node;
    nodes_.insert(node);
  }

 private:
  std::set<Node *> nodes_;
};

// Tell whether Node a links to b.
bool IsNodesLink(Node *a, Node *b) {
  for (auto *node : a->outlinks) {
    if (b == node) {
      return true;
    }
  }
  return false;
}

std::vector<PatternMatcher::subgraph_t> XPUPatternMatcher::DetectPatterns() {
  // Init empty subgraphs.
  std::vector<PatternMatcher::subgraph_t> result;
  std::vector<HitGroup> init_groups;
  std::array<std::vector<HitGroup>, 2> bi_records;
  auto *first_pnode = pattern_.edges().empty() ? pattern().nodes().front().get()
                                               : pattern_.edges().front().first;
  if (!pmnodes2nodes_.count(first_pnode)) return result;
  for (auto *node : pmnodes2nodes_[first_pnode]) {
    HitGroup group;
    group.roles[first_pnode] = node;
    init_groups.emplace_back(group);
  }

  int step = 0;
  bi_records[0] = std::move(init_groups);

  // Extend a PMNode to subgraphs by deducing the connection relations defined
  // in edges of PMNodes.
  for (const auto &edge : pattern_.edges()) {
    VLOG(4) << "check " << edge.first->name() << " -> " << edge.second->name();
    // TODO(Superjomn) Fix bug here, the groups might be duplicate here.
    // Each role has two PMNodes, which indicates two roles.
    // Detect two Nodes that can match these two roles and they are connected.
    auto &pre_groups = bi_records[step % 2];
    auto &cur_groups = bi_records[1 - (step++ % 2)];
    cur_groups.clear();
    if (pre_groups.empty()) break;
    // source -> target
    for (Node *source : pmnodes2nodes_[edge.first]) {
      for (Node *target : pmnodes2nodes_[edge.second]) {
        // TODO(Superjomn) add some prune strategies.
        for (const auto &group : pre_groups) {
          if (IsNodesLink(source, target)) {
            HitGroup new_group = group;
            bool flag = new_group.Match(source, edge.first) &&
                        new_group.Match(target, edge.second);
            if (flag) {
              new_group.Register(source, edge.first);
              new_group.Register(target, edge.second);
              cur_groups.push_back(new_group);
              // TODO(Superjomn) need to unique
            }
          }
        }
      }
    }
    VLOG(3) << "step " << step << " get records: " << cur_groups.size();
  }

  for (auto &group : bi_records[step % 2]) {
    XPUPatternMatcher::subgraph_t subgraph;
    for (auto &role : group.roles) {
      subgraph.emplace(role.first, role.second);
    }
    result.emplace_back(subgraph);
  }
  return result;
}

struct GraphItemLessThan {
  bool operator()(const std::pair<PMNode *, Node *> &a,
                  const std::pair<PMNode *, Node *> &b) {
    if (a.first != b.first) {
      return a.first < b.first;
    } else {
      return a.second < b.second;
    }
  }
};

// TODO(Superjomn) enhance the function as it marks unique unique as duplicates
// see https://github.com/PaddlePaddle/Paddle/issues/13550
void XPUPatternMatcher::UniquePatterns(
    std::vector<PatternMatcher::subgraph_t> *subgraphs) {
  if (subgraphs->empty()) return;
  std::vector<PatternMatcher::subgraph_t> result;

  std::set<size_t> set;
  std::hash<std::string> hasher;
  for (auto &g : *subgraphs) {
    // Sort the items in the sub-graph, and transform to a string key.
    std::vector<std::pair<PMNode *, Node *>> sorted_keys(g.begin(), g.end());
    std::stable_sort(
        sorted_keys.begin(), sorted_keys.end(), GraphItemLessThan());
    STL::stringstream ss;
    for (auto &item : sorted_keys) {
      ss << reinterpret_cast<size_t>(item.first) << ":"
         << reinterpret_cast<size_t>(item.second);
    }
    auto key = hasher(ss.str());
    if (!set.count(key)) {
      result.emplace_back(g);
      set.insert(key);
    }
  }
  *subgraphs = result;
}

void XPUPatternMatcher::RemoveOverlappedMatch(
    std::vector<subgraph_t> *subgraphs) {
  std::vector<subgraph_t> result;
  std::set<Node *> node_set;

  for (const auto &subgraph : *subgraphs) {
    bool valid = true;
    for (auto &item : subgraph) {
      if (item.first->IsIntermediate() && node_set.count(item.second)) {
        valid = false;
        break;
      }
    }
    if (valid) {
      for (auto &item : subgraph) {
        node_set.insert(item.second);
      }
      result.push_back(subgraph);
    }
  }
  *subgraphs = result;
}

}  // namespace xpu
}  // namespace mir
}  // namespace lite
}  // namespace paddle
