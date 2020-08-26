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

#include "lite/core/mir/ssa_graph.h"
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <utility>

namespace paddle {
namespace lite {
namespace mir {

bool SSAGraph::CheckBidirectionalConnection() {
  VLOG(4) << "node count " << node_storage_.size();
  for (auto &node : node_storage_) {
    if (node.IsStmt()) VLOG(6) << node.AsStmt().op_info()->Type();
    if (node.IsArg()) VLOG(6) << node.AsArg().name << " " << node.AsArg().id;
    for (auto *in : node.inlinks) {
      CHECK(in->outlinks.end() !=
            std::find(in->outlinks.begin(), in->outlinks.end(), &node));
    }
    for (auto *out : node.outlinks) {
      CHECK(out->inlinks.end() !=
            std::find(out->inlinks.begin(), out->inlinks.end(), &node));
    }
  }
  return true;
}

std::map<mir::Node *, std::set<mir::Node *>> SSAGraph::BuildOperationAdjList() {
  std::map<mir::Node *, std::set<mir::Node *>> adj_list;

  for (auto &n : mutable_nodes()) {
    if (!n.IsStmt()) continue;
    if (adj_list.find(&n) == adj_list.end()) {
      adj_list[&n] = std::set<mir::Node *>();
    }
    std::vector<mir::Node *> nodes;
    for (auto &var : n.inlinks) {
      for (auto &adj_n : var->inlinks) {
        CHECK(adj_n->IsStmt());
        nodes.push_back(adj_n);
      }
    }
    std::stable_sort(
        nodes.begin(), nodes.end(), [](mir::Node *node1, mir::Node *node2) {
          return node1 > node2;
        });
    adj_list[&n].insert(std::make_move_iterator(nodes.begin()),
                        std::make_move_iterator(nodes.end()));
  }
  return adj_list;
}

std::map<mir::Node *, std::set<mir::Node *>> SSAGraph::BuildNodeAdjList() {
  std::map<mir::Node *, std::set<mir::Node *>> adj_list;

  for (auto &n : mutable_nodes()) {
    if (adj_list.find(&n) == adj_list.end()) {
      adj_list[&n] = std::set<mir::Node *>();
    }
    std::vector<mir::Node *> nodes;
    for (auto &var : n.inlinks) {
      nodes.push_back(var);
    }
    std::stable_sort(
        nodes.begin(), nodes.end(), [](mir::Node *node1, mir::Node *node2) {
          return node1 > node2;
        });
    adj_list[&n].insert(std::make_move_iterator(nodes.begin()),
                        std::make_move_iterator(nodes.end()));
  }
  return adj_list;
}

void SSAGraph::SortHelper(
    const std::map<mir::Node *, std::set<mir::Node *>> &adj_list,
    mir::Node *node,
    std::set<mir::Node *> *visited,
    std::vector<mir::Node *> *ret) {
  visited->insert(node);

  for (auto adj : adj_list.at(node)) {
    if (visited->find(adj) == visited->end()) {
      SortHelper(adj_list, adj, visited, ret);
    }
  }

  ret->push_back(node);
}

std::vector<mir::Node *> SSAGraph::StmtTopologicalOrder() {
  CheckBidirectionalConnection();

  std::stack<mir::Node *> stack;
  std::set<mir::Node *> visited;
  std::vector<mir::Node *> res;

  auto adj_list = BuildOperationAdjList();

  for (auto adj : adj_list) {
    if (visited.find(adj.first) == visited.end()) {
      SortHelper(adj_list, adj.first, &visited, &res);
    }
  }

  return res;
}

std::vector<mir::Node *> SSAGraph::NodeTopologicalOrder() {
  CheckBidirectionalConnection();

  std::stack<mir::Node *> stack;
  std::set<mir::Node *> visited;
  std::vector<mir::Node *> res;

  auto adj_list = BuildNodeAdjList();

  for (auto adj : adj_list) {
    if (visited.find(adj.first) == visited.end()) {
      SortHelper(adj_list, adj.first, &visited, &res);
    }
  }

  return res;
}

Node *SSAGraph::GraphCreateInstructNode(
    const std::shared_ptr<OpLite> &op, const std::vector<Place> &valid_places) {
  node_storage_.emplace_back();
  // TODO(Superjomn) remove one valid_places here.
  op->SetValidPlaces(valid_places);
  auto &new_node = node_storage_.back();
  auto kernels = op->CreateKernels(valid_places);
  node_storage_.back().AsStmt(op->op_type_, std::move(kernels), op);

  CHECK(new_node.inlinks.empty()) << "duplicate Build found";
  CHECK(new_node.outlinks.empty()) << "duplicate Build found";
  return &node_storage_.back();
}

void SSAGraph::Build(const Program &program,
                     const std::vector<Place> &valid_places,
                     int block_idx) {
  CHECK(node_storage_.empty());

  auto weights = program.weights();
  auto is_weight = [&](const std::string &name) -> bool {
    auto it = std::find(weights.begin(), weights.end(), name);
    if (it == weights.end()) return false;
    return true;
  };

  auto var_type_map = program.var_type_map();
  std::map<std::string, mir::Node *> arg_update_node_map;
  for (auto &op : program.ops(block_idx)) {
    VLOG(3) << op->op_info()->Type();
    auto *op_node = GraphCreateInstructNode(op, valid_places);
    auto *op_info = op->op_info();
    const auto &op_type = op_info->Type();
    for (const auto &var_name : op_info->input_names()) {
      mir::Node *arg_node = nullptr;
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
      CHECK(arg_node->IsRoleSet());
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
      CHECK(arg_node->IsRoleSet());
      DirectedLink(op_node, arg_node);
    }
    CHECK(CheckLinksRoleSet());
  }

  CHECK(CheckNodesRoleSet());
  CheckValid();
}

void SSAGraph::RemoveNode(const mir::Node *node) {
  auto pos = std::find_if(node_storage_.begin(),
                          node_storage_.end(),
                          [&node](mir::Node &n) { return &n == node; });
  CHECK(pos != node_storage_.end());
  node_storage_.erase(pos);
}

void SSAGraph::CloneFrom(const SSAGraph &from) {
  node_storage_.clear();
  arguments_.clear();
  valid_places_ = from.valid_places_;

  std::map<const mir::Node *, mir::Node *> clone_node_map;
  for (const auto &node : from.node_storage_) {
    if (node.IsArg()) {
      node_storage_.emplace_back();
      auto &new_node = node_storage_.back();
      new_node.AsArg() = *node.arg();
      clone_node_map.emplace(&node, &new_node);
    } else {
      const auto *inst = node.stmt();
      auto *new_node = GraphCreateInstructNode(inst->op(), valid_places_);
      clone_node_map.emplace(&node, new_node);
    }
  }

  // Rebuild node inlinks/outlinks
  for (const auto &node : from.node_storage_) {
    CHECK(clone_node_map.count(&node));
    auto *new_node = clone_node_map.at(&node);
    for (const auto *inlink : node.inlinks) {
      CHECK(clone_node_map.count(inlink));
      new_node->inlinks.emplace_back(clone_node_map.at(inlink));
    }
    for (const auto *outlink : node.outlinks) {
      CHECK(clone_node_map.count(outlink));
      new_node->outlinks.emplace_back(clone_node_map.at(outlink));
    }
  }

  CheckValid();
}

mir::Node *SSAGraph::Argument(const std::string &name) {
  auto it = arguments_.find(name);
  CHECK(it != arguments_.end()) << "no argument called " << name;
  return it->second;
}

std::vector<mir::Node *> SSAGraph::inputs() {
  std::vector<mir::Node *> res;
  for (auto &node : node_storage_) {
    if (node.inlinks.empty()) {
      res.push_back(&node);
    }
  }
  return res;
}

std::vector<mir::Node *> SSAGraph::outputs() {
  std::vector<mir::Node *> res;
  for (auto &node : node_storage_) {
    if (node.outlinks.empty()) {
      res.push_back(&node);
    }
  }
  return res;
}

mir::Node *SSAGraph::RetrieveArgument(const std::string &arg) {
  for (auto &node : node_storage_) {
    if (node.IsArg() && node.arg()->name == arg) {
      return &node;
    }
  }
  return nullptr;
}

bool SSAGraph::CheckNodesRoleSet() {
  for (auto &node : mutable_nodes()) {
    CHECK_OR_FALSE(node.IsRoleSet());
  }
  return true;
}

bool SSAGraph::CheckLinksRoleSet() {
  for (auto &node : mutable_nodes()) {
    CHECK_OR_FALSE(node.IsRoleSet());
    if (!node.IsStmt()) continue;
    for (auto *x : node.inlinks) {
      CHECK_OR_FALSE(x->IsRoleSet());
      CHECK_OR_FALSE(x->IsArg());
    }
    for (auto *x : node.outlinks) {
      CHECK_OR_FALSE(x->IsRoleSet());
      CHECK_OR_FALSE(x->IsArg());
    }
  }
  return true;
}

Node *SSAGraph::NewArgumentNode(const std::string &name) {
  node_storage_.emplace_back();
  auto &arg_node = node_storage_.back();
  arg_node.AsArg(name, node_storage_.size() - 1);
  return &arg_node;
}

Node *SSAGraph::NewInstructNode() {
  node_storage_.emplace_back();
  return &node_storage_.back();
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
