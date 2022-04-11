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

#include "utility/graph.h"
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <utility>

namespace nnadapter {

Graph::~Graph() { node_storage_.clear(); }

bool Graph::CheckBidirectionalConnection() {
  NNADAPTER_VLOG(5) << "node count " << node_storage_.size();
  for (auto &node : node_storage_) {
    if (node.IsOperation())
      NNADAPTER_VLOG(5) << "Operation "
                        << OperationTypeToString(node.AsOperation().type);
    if (node.IsOperand())
      NNADAPTER_VLOG(5) << "Operand " << OperandToString(&node.AsOperand());
    for (auto *in : node.inlinks) {
      NNADAPTER_CHECK(in->outlinks.end() != std::find(in->outlinks.begin(),
                                                      in->outlinks.end(),
                                                      &node));
    }
    for (auto *out : node.outlinks) {
      NNADAPTER_CHECK(out->inlinks.end() != std::find(out->inlinks.begin(),
                                                      out->inlinks.end(),
                                                      &node));
    }
  }
  return true;
}

std::map<Node *, std::set<Node *>> Graph::BuildOperationAdjList() {
  std::map<Node *, std::set<Node *>> adj_list;

  for (auto &n : mutable_nodes()) {
    if (!n.IsOperation()) continue;
    if (adj_list.find(&n) == adj_list.end()) {
      adj_list[&n] = std::set<Node *>();
    }
    std::vector<Node *> nodes;
    for (auto &var : n.inlinks) {
      for (auto &adj_n : var->inlinks) {
        NNADAPTER_CHECK(adj_n->IsOperation());
        nodes.push_back(adj_n);
      }
    }
    std::stable_sort(nodes.begin(), nodes.end(), [](Node *node1, Node *node2) {
      return node1 > node2;
    });
    adj_list[&n].insert(std::make_move_iterator(nodes.begin()),
                        std::make_move_iterator(nodes.end()));
  }
  return adj_list;
}

std::map<Node *, std::set<Node *>> Graph::BuildNodeAdjList() {
  std::map<Node *, std::set<Node *>> adj_list;

  for (auto &n : mutable_nodes()) {
    if (adj_list.find(&n) == adj_list.end()) {
      adj_list[&n] = std::set<Node *>();
    }
    std::vector<Node *> nodes;
    for (auto &var : n.inlinks) {
      nodes.push_back(var);
    }
    std::stable_sort(nodes.begin(), nodes.end(), [](Node *node1, Node *node2) {
      return node1 > node2;
    });
    adj_list[&n].insert(std::make_move_iterator(nodes.begin()),
                        std::make_move_iterator(nodes.end()));
  }
  return adj_list;
}

void Graph::SortHelper(const std::map<Node *, std::set<Node *>> &adj_list,
                       Node *node,
                       std::set<Node *> *visited,
                       std::vector<Node *> *ret) {
  visited->insert(node);

  for (auto adj : adj_list.at(node)) {
    if (visited->find(adj) == visited->end()) {
      SortHelper(adj_list, adj, visited, ret);
    }
  }

  ret->push_back(node);
}

std::vector<Node *> Graph::OperationTopologicalOrder() {
  CheckBidirectionalConnection();

  std::stack<Node *> stack;
  std::set<Node *> visited;
  std::vector<Node *> res;

  auto adj_list = BuildOperationAdjList();

  for (auto adj : adj_list) {
    if (visited.find(adj.first) == visited.end()) {
      SortHelper(adj_list, adj.first, &visited, &res);
    }
  }

  return res;
}

std::vector<Node *> Graph::NodeTopologicalOrder() {
  CheckBidirectionalConnection();

  std::stack<Node *> stack;
  std::set<Node *> visited;
  std::vector<Node *> res;

  auto adj_list = BuildNodeAdjList();

  for (auto adj : adj_list) {
    if (visited.find(adj.first) == visited.end()) {
      SortHelper(adj_list, adj.first, &visited, &res);
    }
  }

  return res;
}

Node *Graph::GraphCreateInstructNode(core::Operation &op) {
  node_storage_.emplace_back();
  auto &new_node = node_storage_.back();
  node_storage_.back().AsOperation(op);

  NNADAPTER_CHECK(new_node.inlinks.empty()) << "duplicate Build found";
  NNADAPTER_CHECK(new_node.outlinks.empty()) << "duplicate Build found";
  return &node_storage_.back();
}

void Graph::Build(core::Model *model) {
  NNADAPTER_CHECK(node_storage_.empty());
  auto operations = SortOperationsInTopologicalOrder(model);
  std::map<core::Operand *, Node *> arg_update_node_map;
  for (auto *operation : operations) {
    auto *op_node = GraphCreateInstructNode(*operation);
    std::vector<core::Operand *> input_operands = operation->input_operands;
    std::vector<core::Operand *> output_operands = operation->output_operands;
    for (auto *input_operand : input_operands) {
      Node *arg_node = nullptr;
      if (arg_update_node_map.count(input_operand)) {
        arg_node = arg_update_node_map.at(input_operand);
      } else {
        node_storage_.emplace_back();
        arg_node = &node_storage_.back();
        arg_node->AsOperand(*input_operand);
        arg_update_node_map[input_operand] = arg_node;
      }
      NNADAPTER_CHECK(arg_node->IsOperand());
      DirectedLink(arg_node, op_node);
    }
    for (auto *output_operand : output_operands) {
      node_storage_.emplace_back();
      auto *arg_node = &node_storage_.back();
      arg_node->AsOperand(*output_operand);
      arg_update_node_map[output_operand] = arg_node;
      NNADAPTER_CHECK(arg_node->IsOperand());
      DirectedLink(op_node, arg_node);
    }
    NNADAPTER_CHECK(CheckLinksRoleSet());
  }
  NNADAPTER_CHECK(CheckNodesRoleSet());
  CheckValid();
}

void Graph::RemoveNode(const Node *node) {
  auto pos = std::find_if(node_storage_.begin(),
                          node_storage_.end(),
                          [&node](Node &n) { return &n == node; });
  NNADAPTER_CHECK(pos != node_storage_.end());
  node_storage_.erase(pos);
}

void Graph::CloneFrom(const Graph &from) {
  node_storage_.clear();

  std::map<const Node *, Node *> clone_node_map;
  for (const auto &node : from.node_storage_) {
    if (node.IsOperand()) {
      node_storage_.emplace_back();
      auto &new_node = node_storage_.back();
      new_node.AsOperand(*node.operand());
      clone_node_map.emplace(&node, &new_node);
    } else {
      auto *operation = node.operation();
      auto *new_node = GraphCreateInstructNode(*operation);
      clone_node_map.emplace(&node, new_node);
    }
  }

  // Rebuild node inlinks/outlinks
  for (const auto &node : from.node_storage_) {
    NNADAPTER_CHECK(clone_node_map.count(&node));
    auto *new_node = clone_node_map.at(&node);
    for (const auto *inlink : node.inlinks) {
      NNADAPTER_CHECK(clone_node_map.count(inlink));
      new_node->inlinks.emplace_back(clone_node_map.at(inlink));
    }
    for (const auto *outlink : node.outlinks) {
      NNADAPTER_CHECK(clone_node_map.count(outlink));
      new_node->outlinks.emplace_back(clone_node_map.at(outlink));
    }
  }

  CheckValid();
}

std::vector<Node *> Graph::inputs() {
  std::vector<Node *> res;
  for (auto &node : node_storage_) {
    if (node.inlinks.empty()) {
      res.push_back(&node);
    }
  }
  return res;
}

std::vector<Node *> Graph::outputs() {
  std::vector<Node *> res;
  for (auto &node : node_storage_) {
    if (node.outlinks.empty()) {
      res.push_back(&node);
    }
  }
  return res;
}

bool Graph::CheckNodesRoleSet() {
  for (auto &node : mutable_nodes()) {
    NNADAPTER_CHECK(node.IsRoleSet());
  }
  return true;
}

bool Graph::CheckLinksRoleSet() {
  for (auto &node : mutable_nodes()) {
    NNADAPTER_CHECK(node.IsRoleSet());
    if (!node.IsOperation()) continue;
    for (auto *x : node.inlinks) {
      NNADAPTER_CHECK(x->IsRoleSet());
      NNADAPTER_CHECK(x->IsOperand());
    }
    for (auto *x : node.outlinks) {
      NNADAPTER_CHECK(x->IsRoleSet());
      NNADAPTER_CHECK(x->IsOperand());
    }
  }
  return true;
}

Node *Graph::NewArgumentNode() {
  node_storage_.emplace_back();
  auto &arg_node = node_storage_.back();
  arg_node.AsOperand();
  return &arg_node;
}

Node *Graph::NewInstructNode() {
  node_storage_.emplace_back();
  return &node_storage_.back();
}

}  // namespace nnadapter
