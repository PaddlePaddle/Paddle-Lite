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
#include "utility/logging.h"
#include "utility/modeling.h"
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <utility>

namespace nnadapter {

bool Graph::CheckBidirectionalConnection() {
  NNADAPTER_VLOG(5) << "node count " << node_storage_.size();
  for (auto &node : node_storage_) {
    if (node.IsOperation()) NNADAPTER_VLOG(5) << "Operation " << OperationIdToString(node.AsOperation());
    if (node.IsOperand()) NNADAPTER_VLOG(5) << "Operand " << OperandIdToString(node.AsOperand());
    for (auto *in : node.inlinks) {
      NNADAPTER_CHECK(in->outlinks.end() !=
            std::find(in->outlinks.begin(), in->outlinks.end(), &node));
    }
    for (auto *out : node.outlinks) {
      NNADAPTER_CHECK(out->inlinks.end() !=
            std::find(out->inlinks.begin(), out->inlinks.end(), &node));
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
    std::stable_sort(
        nodes.begin(), nodes.end(), [](Node *node1, Node *node2) {
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
    std::stable_sort(
        nodes.begin(), nodes.end(), [](Node *node1, Node *node2) {
          return node1 > node2;
        });
    adj_list[&n].insert(std::make_move_iterator(nodes.begin()),
                        std::make_move_iterator(nodes.end()));
  }
  return adj_list;
}

void Graph::SortHelper(
    const std::map<Node *, std::set<Node *>> &adj_list,
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

Node *Graph::GraphCreateInstructNode(const core::Operation &op) {
  node_storage_.emplace_back();
  auto &new_node = node_storage_.back();
  node_storage_.back().AsOperation(op);

  NNADAPTER_CHECK(new_node.inlinks.empty()) << "duplicate Build found";
  NNADAPTER_CHECK(new_node.outlinks.empty()) << "duplicate Build found";
  return &node_storage_.back();
}

void Graph::Build(const core::Model) {
  NNADAPTER_CHECK(node_storage_.empty());

  block_idx_ = block_idx;
  auto weights = program.weights();
  auto is_weight = [&](const std::string &name) -> bool {
    auto it = std::find(weights.begin(), weights.end(), name);
    if (it == weights.end()) return false;
    return true;
  };

  auto var_type_map = program.var_type_map();
  std::map<std::string, Node *> arg_update_node_map;
  for (auto &op : program.ops(block_idx)) {
    NNADAPTER_VLOG(3) << op->op_info()->Type();
    auto *op_node = GraphCreateInstructNode(op, valid_places);
    auto *op_info = op->op_info();
    const auto &op_type = op_info->Type();
    for (const auto &var_name : op_info->input_names()) {
      Node *arg_node = nullptr;
      if (arg_update_node_map.count(var_name)) {
        arg_node = arg_update_node_map.at(var_name);
      } else {
        node_storage_.emplace_back();
        arg_node = &node_storage_.back();
        arg_node->AsArg(var_name, node_storage_.size() - 1);
        arg_update_node_map[var_name] = arg_node;
      }
      if (var_type_map.count(var_name)) {
        if (!arg_node->arg()->type) {
          arg_node->arg()->type = var_type_map[var_name];
        }
        // Store the original data type of the output tensors for
        // type_precision_cast_pass, to keep the consistency between the
        // output types of original graph and optimized graph's
        if (op_type == "fetch") {
          op->mutable_op_info()->SetAttr<int>(
              "data_type",
              static_cast<int>(var_type_map[var_name]->precision()));
        }
      }
      if (is_weight(var_name)) arg_node->AsArg().is_weight = true;
      NNADAPTER_CHECK(arg_node->IsRoleSet());
      DirectedLink(arg_node, op_node);
    }
    for (const auto &var_name : op->op_info()->output_names()) {
      node_storage_.emplace_back();
      auto *arg_node = &node_storage_.back();
      arg_node->AsArg(var_name, node_storage_.size() - 1);
      arg_update_node_map[var_name] = arg_node;
      if (var_type_map.count(var_name) && !arg_node->arg()->type) {
        arg_node->arg()->type = var_type_map[var_name];
      }

      if (is_weight(var_name)) arg_node->AsArg().is_weight = true;
      NNADAPTER_CHECK(arg_node->IsRoleSet());
      DirectedLink(op_node, arg_node);
    }
    NNADAPTER_CHECK(NNADAPTER_CHECKLinksRoleSet());
  }

  NNADAPTER_CHECK(NNADAPTER_CHECKNodesRoleSet());
  NNADAPTER_CHECKValid();
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
  arguments_.clear();

  std::map<const Node *, Node *> clone_node_map;
  for (const auto &node : from.node_storage_) {
    if (node.IsOperand()) {
      node_storage_.emplace_back();
      auto &new_node = node_storage_.back();
      new_node.AsOperand() = node.operand();
      clone_node_map.emplace(&node, &new_node);
    } else {
      const auto *operation = node.operation();
      auto *new_node = GraphCreateInstructNode(operation);
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

  NNADAPTER_CHECKValid();
}

Node *Graph::Argument(const std::string &name) {
  auto it = arguments_.find(name);
  NNADAPTER_CHECK(it != arguments_.end()) << "no argument called " << name;
  return it->second;
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

Node *Graph::RetrieveArgument(const std::string &arg) {
  for (auto &node : node_storage_) {
    if (node.IsOperand() && node.arg()->name == arg) {
      return &node;
    }
  }
  return nullptr;
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

Node *Graph::NewArgumentNode(const std::string &name) {
  node_storage_.emplace_back();
  auto &arg_node = node_storage_.back();
  arg_node.AsArg(name, node_storage_.size() - 1);
  return &arg_node;
}

Node *Graph::NewInstructNode() {
  node_storage_.emplace_back();
  return &node_storage_.back();
}

}  // namespace nnadapter
