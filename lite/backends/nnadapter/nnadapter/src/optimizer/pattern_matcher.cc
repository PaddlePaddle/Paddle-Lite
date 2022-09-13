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

#include "optimizer/pattern_matcher.h"
#include <algorithm>
#include <array>
#include "utility/debug.h"
#include "utility/micros.h"
#include "utility/modeling.h"

namespace nnadapter {

NNADAPTER_EXPORT bool PatternMatcher::Node::IsOperand() const {
  NNADAPTER_CHECK(operand || operation);
  return operand && !operation;
}

NNADAPTER_EXPORT bool PatternMatcher::Node::IsOperation() const {
  NNADAPTER_CHECK(operand || operation);
  return !operand && operation;
}

NNADAPTER_EXPORT PatternMatcher::Pattern::Pattern(std::vector<Edge> *e,
                                                  NNAdapterOperationType t)
    : edges(e), type(t) {
  if (type != NNADAPTER_UNKNOWN) {
    IsOperation(type);
  }
}

NNADAPTER_EXPORT PatternMatcher::Pattern &PatternMatcher::Pattern::operator>>(
    Pattern &other) {
  NNADAPTER_CHECK(this != &other) << "Can't link to the same pattern.";
  edges->emplace_back(this, &other);
  // Add out operation link automatically
  if (other.type != NNADAPTER_UNKNOWN) {
    IsOperationInputOperand(other.type);
  }
  return other;
}

NNADAPTER_EXPORT PatternMatcher::Pattern &PatternMatcher::Pattern::operator>>(
    std::vector<Pattern *> &others) {
  for (auto other : others) {
    *this >> *other;
  }
  return *this;
}

NNADAPTER_EXPORT PatternMatcher::Pattern &operator>>(
    std::vector<PatternMatcher::Pattern *> &others,
    PatternMatcher::Pattern &self) {
  for (auto other : others) {
    *other >> self;
  }
  return self;
}

NNADAPTER_EXPORT bool PatternMatcher::Pattern::MatchAllConditions(
    const Node *node) const {
  for (auto &condition : conditions) {
    if (!condition(node)) return false;
  }
  return true;
}

NNADAPTER_EXPORT PatternMatcher::Pattern *PatternMatcher::Pattern::IsOperand() {
  conditions.emplace_back(
      [](const Node *node) { return node && node->IsOperand(); });
  return this;
}

NNADAPTER_EXPORT PatternMatcher::Pattern *
PatternMatcher::Pattern::IsConstantOperand() {
  IsOperand();
  conditions.emplace_back([=](const Node *node) {
    return nnadapter::IsConstantOperand(node->operand);
  });
  return this;
}

NNADAPTER_EXPORT PatternMatcher::Pattern *
PatternMatcher::Pattern::IsVariableOperand() {
  IsOperand();
  conditions.emplace_back([=](const Node *node) {
    return !nnadapter::IsConstantOperand(node->operand);
  });
  return this;
}

NNADAPTER_EXPORT PatternMatcher::Pattern *
PatternMatcher::Pattern::IsConstantCopyOperand() {
  IsOperand();
  conditions.emplace_back([=](const Node *node) {
    return nnadapter::IsConstantCopyOperand(node->operand);
  });
  return this;
}

NNADAPTER_EXPORT PatternMatcher::Pattern *
PatternMatcher::Pattern::IsConstantReferenceOperand() {
  IsOperand();
  conditions.emplace_back([=](const Node *node) {
    return nnadapter::IsConstantReferenceOperand(node->operand);
  });
  return this;
}

NNADAPTER_EXPORT PatternMatcher::Pattern *
PatternMatcher::Pattern::IsTemporaryVariableOperand() {
  IsOperand();
  conditions.emplace_back([=](const Node *node) {
    return nnadapter::IsTemporaryVariableOperand(node->operand);
  });
  return this;
}

NNADAPTER_EXPORT PatternMatcher::Pattern *
PatternMatcher::Pattern::IsTemporaryShapeOperand() {
  IsOperand();
  conditions.emplace_back([=](const Node *node) {
    return nnadapter::IsTemporaryShapeOperand(node->operand);
  });
  return this;
}

NNADAPTER_EXPORT PatternMatcher::Pattern *
PatternMatcher::Pattern::IsOperationInputOperand(NNAdapterOperationType type,
                                                 int index) {
  IsOperand();
  conditions.emplace_back([=](const Node *node) {
    for (auto op : node->outlinks) {
      if (op && op->IsOperation() && op->operation->type == type) {
        auto &operands = op->operation->input_operands;
        if (index >= 0) {
          if (operands.size() > index && operands[index] == node->operand) {
            return true;
          }
        } else {
          return true;
        }
      }
    }
    return false;
  });
  return this;
}

NNADAPTER_EXPORT PatternMatcher::Pattern *
PatternMatcher::Pattern::IsOperationOutputOperand(NNAdapterOperationType type,
                                                  int index) {
  IsOperand();
  conditions.emplace_back([=](const Node *node) {
    for (auto op : node->inlinks) {
      if (op && op->IsOperation() && op->operation->type == type) {
        auto &operands = op->operation->output_operands;
        if (index >= 0) {
          if (operands.size() > index && operands[index] == node->operand) {
            return true;
          }
        } else {
          return true;
        }
      }
    }
    return false;
  });
  return this;
}

NNADAPTER_EXPORT PatternMatcher::Pattern *PatternMatcher::Pattern::IsOperation(
    NNAdapterOperationType type) {
  conditions.emplace_back([type](const Node *node) {
    return node && node->IsOperation() &&
           (node->operation->type == type || type == NNADAPTER_UNKNOWN);
  });
  return this;
}

NNADAPTER_EXPORT PatternMatcher::Pattern *
PatternMatcher::Pattern::CheckInputCount(int num) {
  conditions.emplace_back(
      [num](const Node *node) { return node->inlinks.size() == num; });
  return this;
}

NNADAPTER_EXPORT PatternMatcher::Pattern *
PatternMatcher::Pattern::CheckOutputCount(int num) {
  conditions.emplace_back(
      [num](const Node *node) { return node->outlinks.size() == num; });
  return this;
}

NNADAPTER_EXPORT PatternMatcher::Pattern *
PatternMatcher::Pattern::IsIntermediate() {
  intermediate = true;
  return this;
}

NNADAPTER_EXPORT PatternMatcher::Pattern *
PatternMatcher::Pattern::MatchCondition(const Condition &condition) {
  conditions.push_back(condition);
  return this;
}

NNADAPTER_EXPORT PatternMatcher::Pattern *PatternMatcher::CreatePattern(
    const std::string &key, NNAdapterOperationType type) {
  NNADAPTER_CHECK_EQ(patterns_.count(key), 0);
  auto pattern = new Pattern(&edges_, type);
  patterns_[key] = std::move(std::unique_ptr<Pattern>(pattern));
  return pattern;
}

NNADAPTER_EXPORT size_t PatternMatcher::Apply(core::Model *model) {
  BuildPattern();
  std::list<Node> nodes;
  BuildNodes(model, &nodes);
  if (!MarkPatterns(&nodes)) {
    return 0;
  }
  auto subgraphs = DetectPatterns();
  NNADAPTER_VLOG(5) << subgraphs.size() << " subgraphs detected!";
  UniquePatterns(&subgraphs);
  NNADAPTER_VLOG(5) << subgraphs.size() << " subgraphs unique!";
  ValidatePatterns(&subgraphs);
  NNADAPTER_VLOG(5) << subgraphs.size() << " subgraphs valid!";
  RemoveOverlappedPatterns(&subgraphs);
  NNADAPTER_VLOG(5) << subgraphs.size() << " subgraphs matched!";
  // Notify to handle the matched subgraphs, and collect the intermediate
  // operands and operations to be deleted.
  size_t count = 0;
  std::set<core::Operand *> operands;
  std::set<core::Operation *> operations;
  for (auto &subgraph : subgraphs) {
    std::map<std::string, Node *> nodes;
    std::set<Node *> intermediates;
    for (auto &pattern : patterns_) {
      auto node = subgraph.at(pattern.second.get());
      if (pattern.second->intermediate) {
        intermediates.insert(node);
      }
      nodes[pattern.first] = node;
    }
    // If it returns true, add the intermediate operands and operations to the
    // list to be deleted, otherwise ignore it.
    if (!HandleMatchedResults(model, nodes)) continue;
    for (auto intermediate : intermediates) {
      if (intermediate->IsOperand()) {
        operands.insert(intermediate->operand);
      } else if (intermediate->IsOperation()) {
        operations.insert(intermediate->operation);
      } else {
        NNADAPTER_LOG(FATAL)
            << "The intermediate node is neither an operator nor an operation!";
      }
    }
    count++;
  }
  NNADAPTER_VLOG(5) << count << " subgraphs replaced!";
  // Delete the intermediate operands and operations.
  for (auto operand : operands) {
    RemoveOperand(model, operand);
  }
  NNADAPTER_VLOG(3) << operands.size() << " operands deleted!";
  for (auto operation : operations) {
    RemoveOperation(model, operation);
  }
  NNADAPTER_VLOG(3) << operations.size() << " operations deleted!";
  return count;
}

NNADAPTER_EXPORT void PatternMatcher::BuildNodes(core::Model *model,
                                                 std::list<Node> *nodes) {
  nodes->clear();
  std::unordered_map<core::Operand *, Node *> operand2node;
  for (auto &operation : model->operations) {
    nodes->emplace_back(&operation);
    auto operation_node = &nodes->back();
    for (auto operand : operation.input_operands) {
      Node *operand_node = nullptr;
      if (operand) {
        if (!operand2node.count(operand)) {
          nodes->emplace_back(operand);
          operand2node[operand] = &nodes->back();
        }
        operand_node = operand2node[operand];
        operand_node->outlinks.push_back(operation_node);
      }
      operation_node->inlinks.push_back(operand_node);
    }
    for (auto operand : operation.output_operands) {
      Node *operand_node = nullptr;
      if (operand) {
        if (!operand2node.count(operand)) {
          nodes->emplace_back(operand);
          operand2node[operand] = &nodes->back();
        }
        operand_node = operand2node[operand];
        operand_node->inlinks.push_back(operation_node);
      }
      operation_node->outlinks.push_back(operand_node);
    }
  }
  NNADAPTER_VLOG(5) << nodes->size() << " nodes created!";
}

NNADAPTER_EXPORT bool PatternMatcher::MarkPatterns(std::list<Node> *nodes) {
  if (!nodes || nodes->empty()) return false;
  for (auto &node : *nodes) {
    for (const auto &pattern : patterns_) {
      if (pattern.second->MatchAllConditions(&node)) {
        pattern2nodes_[pattern.second.get()].insert(&node);
      }
    }
  }
  // Check to early stop if some patterns can't find the matched nodes.
  for (auto &pattern : patterns_) {
    if (!pattern2nodes_.count(pattern.second.get())) {
      NNADAPTER_VLOG(4) << "Can't find matched pattern, early stop!";
      // return false;
    }
  }
  NNADAPTER_VLOG(5) << pattern2nodes_.size() << " patterns marked!";
  return !pattern2nodes_.empty();
}

struct HitGroup {
  std::map<PatternMatcher::Pattern *, PatternMatcher::Node *> roles;
  bool Match(PatternMatcher::Node *node, PatternMatcher::Pattern *pattern) {
    if (nodes.count(node)) {
      if (roles.count(pattern) && roles[pattern] == node) return true;
      return false;
    } else {
      if (roles.count(pattern) && roles[pattern] != node) return false;
      return true;
    }
  }
  void Register(PatternMatcher::Node *node, PatternMatcher::Pattern *pattern) {
    roles[pattern] = node;
    nodes.insert(node);
  }

 private:
  std::set<PatternMatcher::Node *> nodes;
};

NNADAPTER_EXPORT std::vector<PatternMatcher::Subgraph>
PatternMatcher::DetectPatterns() {
  // Init empty subgraphs
  std::vector<PatternMatcher::Subgraph> result;
  std::vector<HitGroup> init_groups;
  std::array<std::vector<HitGroup>, 2> bi_records;
  auto first_pattern =
      edges_.empty() ? patterns_.begin()->second.get() : edges_.front().first;
  if (!pattern2nodes_.count(first_pattern)) return result;
  for (auto node : pattern2nodes_[first_pattern]) {
    HitGroup group;
    group.roles[first_pattern] = node;
    init_groups.emplace_back(group);
  }
  int step = 0;
  bi_records[0] = std::move(init_groups);
  // Extend a pattern to subgraphs by deducing the connection relations defined
  // in edges of patterns
  for (const auto &edge : edges_) {
    // TODO(Superjomn) Fix bug here, the groups might be duplicate here.
    // Each role has two patterns, which indicates two roles.
    // Detect two nodes that can match these two roles and they are connected.
    auto &pre_groups = bi_records[step % 2];
    auto &cur_groups = bi_records[1 - (step++ % 2)];
    cur_groups.clear();
    if (pre_groups.empty()) break;
    // source -> target
    for (auto source : pattern2nodes_[edge.first]) {
      for (auto target : pattern2nodes_[edge.second]) {
        // TODO(Superjomn) add some prune strategies.
        for (const auto &group : pre_groups) {
          bool linked = false;
          for (auto node : source->outlinks) {
            if (target == node) {
              linked = true;
              break;
            }
          }
          if (linked) {
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
    NNADAPTER_VLOG(5) << step << "th step get records: " << cur_groups.size();
  }
  for (auto &group : bi_records[step % 2]) {
    Subgraph subgraph;
    for (auto &role : group.roles) {
      subgraph.emplace(role.first, role.second);
    }
    result.emplace_back(subgraph);
  }
  return result;
}

// TODO(Superjomn) enhance the function as it marks unique unique as duplicates
// see https://github.com/PaddlePaddle/Paddle/issues/13550
NNADAPTER_EXPORT void PatternMatcher::UniquePatterns(
    std::vector<PatternMatcher::Subgraph> *subgraphs) {
  if (subgraphs->empty()) return;
  std::vector<PatternMatcher::Subgraph> result;
  std::set<size_t> hash_keys;
  std::hash<std::string> hasher;
  for (auto &subgraph : *subgraphs) {
    // Sort the items in the sub-graph, and transform to a string key.
    std::vector<std::pair<Pattern *, Node *>> sorted_keys(subgraph.begin(),
                                                          subgraph.end());
    std::stable_sort(sorted_keys.begin(),
                     sorted_keys.end(),
                     [](const std::pair<Pattern *, Node *> &a,
                        const std::pair<Pattern *, Node *> &b) {
                       if (a.first != b.first) {
                         return a.first < b.first;
                       } else {
                         return a.second < b.second;
                       }
                     });
    std::stringstream ss;
    for (auto &sorted_key : sorted_keys) {
      ss << reinterpret_cast<size_t>(sorted_key.first) << ":"
         << reinterpret_cast<size_t>(sorted_key.second);
    }
    auto hash_key = hasher(ss.str());
    if (!hash_keys.count(hash_key)) {
      result.emplace_back(subgraph);
      hash_keys.insert(hash_key);
    }
  }
  *subgraphs = result;
}

// The intermediate nodes can only link to the nodes inside the pattern, or the
// subgraph will be droped.
NNADAPTER_EXPORT void PatternMatcher::ValidatePatterns(
    std::vector<Subgraph> *subgraphs) {
  subgraphs->erase(
      std::remove_if(subgraphs->begin(),
                     subgraphs->end(),
                     [](const Subgraph &subgraph) -> bool {
                       // Collect the inlinks and outlinks.
                       std::set<Node *> nodes;
                       for (auto &pattern : subgraph) {
                         nodes.insert(pattern.second);
                       }
                       for (auto &pattern : subgraph) {
                         if (pattern.first->intermediate) {
                           for (auto node : pattern.second->inlinks) {
                             if (!nodes.count(node)) {
                               return true;
                             }
                           }
                           for (auto node : pattern.second->outlinks) {
                             if (!nodes.count(node)) {
                               return true;
                             }
                           }
                         }
                       }
                       return false;
                     }),
      subgraphs->end());
}

NNADAPTER_EXPORT void PatternMatcher::RemoveOverlappedPatterns(
    std::vector<Subgraph> *subgraphs) {
  std::vector<Subgraph> result;
  std::set<Node *> nodes;
  for (const auto &subgraph : *subgraphs) {
    bool valid = true;
    for (auto &pattern : subgraph) {
      if (pattern.first->intermediate && nodes.count(pattern.second)) {
        valid = false;
        break;
      }
    }
    if (valid) {
      for (auto &pattern : subgraph) {
        nodes.insert(pattern.second);
      }
      result.push_back(subgraph);
    }
  }
  *subgraphs = result;
}

}  // namespace nnadapter
