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

#include "utility/pattern_matcher.h"
#include <algorithm>
#include <array>
#include <string>
#include <vector>
#include "utility/debug.h"

namespace nnadapter {

size_t PMPattern::id_ = 0UL;

PMNode &PMNode::operator>>(PMNode &right) {
  pattern_->AddEdge(this, &right);
  // automatically add out op link relation.
  if (right.IsOp()) {
    this->assert_is_op_input(right.op_type_);
  }

  return right;
}

PMNode &PMNode::operator>>(std::vector<PMNode *> &nodes) {
  for (auto *node : nodes) {
    *this >> *node;
  }
  return *this;
}

PMNode &operator>>(std::vector<PMNode *> &others, PMNode &me) {
  for (auto *o : others) {
    *o >> me;
  }
  return me;
}

PMNode *PMPattern::NewNode(const std::string &name) {
  if (!name.empty()) {
    NNADAPTER_CHECK_EQ(node_map_.count(name), 0UL)
        << "PMNode's name should be unique, get duplicate " << name;
  }

  nodes_.emplace_back(new PMNode(this, name));
  auto *cur = nodes_.back().get();
  node_map_[name] = cur;
  return cur;
}

PMNode *PMPattern::NewNode(PMNode::teller_t &&teller, const std::string &name) {
  if (!name.empty()) {
    NNADAPTER_CHECK_EQ(node_map_.count(name), 0UL)
        << "PMNode's name should be unique, get duplicate " << name;
  }

  nodes_.emplace_back(new PMNode(std::move(teller), this, name));
  auto *cur = nodes_.back().get();
  node_map_[name] = cur;
  return cur;
}

PMNode *PMPattern::RetrieveNode(const std::string &id) const {
  auto it = node_map_.find(id);
  if (it == node_map_.end()) {
    return nullptr;
  }

  return it->second;
}

void PMPattern::AddEdge(PMNode *a, PMNode *b) {
  NNADAPTER_CHECK(a);
  NNADAPTER_CHECK(b);
  NNADAPTER_CHECK_NE(a, b) << "Can't connect to the same nodes.";
  edges_.emplace_back(a, b);
}

void PatternMatcher::operator()(Graph *graph,
                                PatternMatcher::handle_t handler) {
  if (!MarkPMNodesInGraph(graph)) {
    return;
  }

  auto subgraphs = DetectPatterns();
  UniquePatterns(&subgraphs);
  ValidateByNodeRole(&subgraphs);
  RemoveOverlappedMatch(&subgraphs);

  if (subgraphs.empty()) return;
  NNADAPTER_LOG(INFO) << "detected " << subgraphs.size() << " subgraph";
  int id = 0;
  for (auto &g : subgraphs) {
    NNADAPTER_VLOG(3) << "optimizing #" << id++ << " subgraph";
    handler(g, graph);
  }
}

bool PatternMatcher::MarkPMNodesInGraph(Graph *graph) {
  NNADAPTER_VLOG(3) << "mark pmnodes in graph";
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
      NNADAPTER_VLOG(4) << pmnode->name()
                        << " can't find matched Node, early stop";
      // return false;
    }
  }
  NNADAPTER_VLOG(3) << pmnodes2nodes_.size() << " nodes marked";

  return !pmnodes2nodes_.empty();
}

// The intermediate Nodes can only link to the nodes inside the pattern, or this
// subgraph will be droped.
void PatternMatcher::ValidateByNodeRole(
    std::vector<PatternMatcher::subgraph_t> *subgraphs) {
  subgraphs->erase(
      std::remove_if(subgraphs->begin(),
                     subgraphs->end(),
                     [](const PatternMatcher::subgraph_t &subgraph) -> bool {
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
        if (x->IsOperand() && ios.count(x) == 0) {
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

std::vector<PatternMatcher::subgraph_t> PatternMatcher::DetectPatterns() {
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
    NNADAPTER_VLOG(4) << "NNADAPTER_CHECK " << edge.first->name() << " -> "
                      << edge.second->name();
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
        for (const auto &group : pre_groups) {
          if (IsNodesLink(source, target)) {
            HitGroup new_group = group;
            bool flag = new_group.Match(source, edge.first) &&
                        new_group.Match(target, edge.second);
            if (flag) {
              new_group.Register(source, edge.first);
              new_group.Register(target, edge.second);
              cur_groups.push_back(new_group);
            }
          }
        }
      }
    }
    NNADAPTER_VLOG(3) << "step " << step
                      << " get records: " << cur_groups.size();
  }

  for (auto &group : bi_records[step % 2]) {
    PatternMatcher::subgraph_t subgraph;
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
void PatternMatcher::UniquePatterns(
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
    std::stringstream ss;
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

void PatternMatcher::RemoveOverlappedMatch(std::vector<subgraph_t> *subgraphs) {
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

std::string PMPattern::DotString() const {
  Dot dot;
  int id = 0;
  // Create Nodes
  std::map<PMNode *, std::string> node2dot;
  for (const auto &node : nodes()) {
    std::string node_id = string_format("Node%d", id++);
    dot.AddNode(node_id, {}, node->name());
    node2dot[node.get()] = node_id;
  }
  // Create Edges
  for (const auto &edge : edges()) {
    if (!node2dot.count(edge.first) || !node2dot.count(edge.second)) {
      continue;
    }
    auto &src = node2dot.at(edge.first);
    auto &trg = node2dot.at(edge.second);
    dot.AddEdge(src, trg, {});
  }
  return dot.Build();
}

PMNode &PMNode::LinksTo(const std::vector<PMNode *> &others) {
  // extend outlinks.
  for (PMNode *x : others) {
    pattern_->AddEdge(this, x);
  }
  return *this;
}

PMNode &PMNode::LinksFrom(const std::vector<PMNode *> &others) {
  // extend outlinks.
  for (PMNode *x : others) {
    pattern_->AddEdge(x, this);
  }
  return *this;
}

PMNode *PMNode::assert_is_op() {
  asserts_.emplace_back([](const Node *x) { return x && x->IsOperation(); });
  return this;
}

PMNode *PMNode::assert_only_one_output() {
  asserts_.emplace_back([](const Node *x) { return x->outlinks.size() == 1; });
  return this;
}

PMNode *PMNode::assert_is_op(NNAdapterOperationType op_type) {
  asserts_.emplace_back([op_type](const Node *x) {
    if (x && x->IsOperation()) {
      auto *operation = x->operation();
      return operation->type == op_type;
    } else {
      return false;
    }
  });
  return this;
}

PMNode *PMNode::assert_is_not_op_type(NNAdapterOperationType op_type) {
  asserts_.emplace_back([op_type](const Node *x) {
    if (x && x->IsOperation()) {
      auto *operation = x->operation();
      if (operation->type == op_type) {
        return false;
      }
    }
    return true;
  });
  return this;
}

PMNode *PMNode::assert_is_var() {
  asserts_.emplace_back([](const Node *x) { return x && x->IsOperand(); });
  return this;
}

PMNode *PMNode::assert_var_not_persistable() {
  assert_is_var();
  asserts_.emplace_back(
      [](const Node *x) { return !IsConstantOperand(x->operand()); });
  return this;
}

PMNode *PMNode::assert_is_persistable_var() {
  assert_is_var();
  asserts_.emplace_back(
      [=](const Node *x) { return IsConstantOperand(x->operand()); });
  return this;
}

PMNode *PMNode::assert_is_op_output(NNAdapterOperationType op_type) {
  assert_is_var();
  asserts_.emplace_back([=](const Node *x) {
    for (auto *op : x->inlinks) {
      if (op && op->IsOperation()) {
        auto *operation = x->operation();
        if (operation->type == op_type) return true;
      }
    }
    return false;
  });
  return this;
}

bool IsNthOutput(const Node *var, const Node *op, size_t nth) {
  NNADAPTER_CHECK(var->IsOperand());
  NNADAPTER_CHECK(op->IsOperation());
  auto operation = op->operation();
  auto output_operands = operation->output_operands;
  if (!std::count(
          output_operands.begin(), output_operands.end(), var->operand()) ||
      output_operands.size() <= nth)
    return false;
  return var->operand() == output_operands[nth];
}

bool IsNthInput(const Node *var, const Node *op, size_t nth) {
  NNADAPTER_CHECK(var->IsOperand());
  NNADAPTER_CHECK(op->IsOperation());
  auto operation = op->operation();
  auto input_operands = operation->input_operands;
  if (!std::count(
          input_operands.begin(), input_operands.end(), var->operand()) ||
      input_operands.size() <= nth)
    return false;
  return var->operand() == input_operands[nth];
}

PMNode *PMNode::assert_is_op_nth_input(NNAdapterOperationType op_type,
                                       int nth) {
  assert_is_var();
  assert_is_op_input(op_type);
  asserts_.emplace_back([=](const Node *x) {
    for (auto *op : x->outlinks) {
      NNADAPTER_LOG(INFO) << *op;
      if (op && op->IsOperation() && op->operation()->type == op_type &&
          IsNthInput(x, op, nth))
        return true;
    }
    return false;
  });
  return this;
}

PMNode *PMNode::assert_is_op_nth_output(NNAdapterOperationType op_type,
                                        int nth) {
  assert_is_var();
  asserts_.emplace_back([=](const Node *x) {
    for (auto *op : x->inlinks) {
      if (op && op->IsOperation() && op->operation()->type == op_type &&
          IsNthOutput(x, op, nth))
        return true;
    }
    return false;
  });
  return this;
}

PMNode *PMNode::assert_is_op_input(NNAdapterOperationType op_type) {
  assert_is_var();
  asserts_.emplace_back([=](const Node *x) {
    for (auto *op : x->outlinks) {
      if (op && op->IsOperation()) {
        auto *operation = op->operation();
        if (operation->type == op_type) {
          return true;
        }
      }
    }
    return false;
  });
  return this;
}

void GraphSafeRemoveNodes(Graph *graph, const std::set<const Node *> &nodes) {
  for (auto *node : nodes) {
    graph->RemoveNode(node);
  }

  for (auto &node : graph->mutable_nodes()) {
    for (auto it = node.inlinks.begin(); it != node.inlinks.end();) {
      if (nodes.count(*it)) {
        it = node.inlinks.erase(it);
      } else {
        it++;
      }
    }
    for (auto it = node.outlinks.begin(); it != node.outlinks.end();) {
      if (nodes.count(*it)) {
        it = node.outlinks.erase(it);
      } else {
        it++;
      }
    }
  }
}

}  // namespace nnadapter
