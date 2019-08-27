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

#include "lite/core/mir/subgraph/subgraph_program_pass.h"
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher.h"

namespace paddle {
namespace lite {
namespace mir {
namespace subgraph {

void SubgraphProgramPass::SortHelper(
    Node* node,
    const std::unordered_set<Node*>& nodes_all,
    std::unordered_set<const Node*>* visited_nodes,
    std::vector<Node*>* ret) {
  for (auto& var_node : node->inlinks) {
    if (var_node->inlinks.empty()) continue;
    auto* op_node = var_node->inlinks.front();
    if (nodes_all.count(op_node) && !visited_nodes->count(op_node)) {
      SortHelper(op_node, nodes_all, visited_nodes, ret);
    }
  }
  ret->push_back(node);
  visited_nodes->insert(node);
}

std::vector<Node*> SubgraphProgramPass::GetTopologicalOrder(
    const std::unordered_set<Node*>& nodes) {
  std::unordered_set<const Node*> visited;
  std::vector<Node*> ret;
  for (auto& node : nodes) {
    if (!node->IsStmt()) continue;
    if (visited.count(node)) continue;
    SortHelper(node, nodes, &visited, &ret);
  }
  return ret;
}

void SubgraphProgramPass::InferOnce(const std::unique_ptr<SSAGraph>& graph) {
  for (auto& item : graph->StmtTopologicalOrder()) {
    if (!item->IsStmt()) continue;
    auto& stmt = item->AsStmt();
    auto& op = stmt.op();
    op->CheckShape();
    op->InferShape();
    // TOOD(xxx): remove Launch() at last
    auto& kkks = stmt.kernels();
    if (!kkks.empty()) {
      auto& kk = stmt.kernels().front();
      if (kk) {
        kk->Launch();
      }
    }
  }
}

void SubgraphProgramPass::InitSubgraphID(
    const std::unique_ptr<SSAGraph>& graph,
    const std::vector<std::string>& supported_op_types) {
  for (auto& item : graph->StmtTopologicalOrder()) {
    if (!item->IsStmt()) continue;
    auto& stmt = item->AsStmt();
    stmt.ClearSubgraphID();
    if (std::find(supported_op_types.begin(),
                  supported_op_types.end(),
                  stmt.op_type()) != supported_op_types.end()) {
      stmt.SetSubgraphID(0);
      LOG(INFO) << "supported " << stmt.op_type();
    } else {
      LOG(INFO) << "======= not supported " << stmt.op_type();
    }
  }
}

// mark current and all output supported nodes
void SubgraphProgramPass::ChangeAllOutConnectedID(Node* node,
                                                  int to_id,
                                                  int from_id) {
  if (!node) return;
  if (node->IsStmt()) {
    auto& stmt = node->AsStmt();
    if (stmt.subgraph_id() == from_id) {
      stmt.SetSubgraphID(to_id);
      nodes2rm_[to_id].insert(node);
      for (auto& i : node->outlinks) {
        ChangeAllOutConnectedID(i, to_id, from_id);
      }
    } else {
      LOG(INFO) << "failed op type:" << stmt.op_type();
      return;
    }
  } else {
    // this it arg node
    bool all_out_op_supported = true;
    for (auto& i : node->outlinks) {
      if (!i->IsStmt()) return;
      auto& stmt = i->AsStmt();
      if (stmt.subgraph_id() < from_id) {
        all_out_op_supported = false;
      }
    }
    if (!all_out_op_supported) {
      return;
    }
    nodes2rm_[to_id].insert(node);
    for (auto& i : node->outlinks) {
      CHECK(i->IsStmt());
      auto& stmt = i->AsStmt();
      if (stmt.subgraph_id() == from_id) {
        stmt.SetSubgraphID(to_id);
        nodes2rm_[to_id].insert(i);
        for (auto& o : i->outlinks) {
          for (auto& j : o->outlinks) {
            if (j->IsStmt()) {
              auto& Nstmt = j->AsStmt();
              if (Nstmt.subgraph_id() < from_id) {
                o_nodes_[to_id].insert(o);
              }
            }
          }
          ChangeAllOutConnectedID(o, to_id, from_id);
        }
      }
    }
  }
}

int SubgraphProgramPass::FuseSubgraphID(
    const std::unique_ptr<SSAGraph>& graph) {
  int sub_id = 1;  // id start from 1 not 0
  for (auto& item : graph->StmtTopologicalOrder()) {
    bool inputvar = 0;
    if (!item->IsStmt()) continue;
    auto& stmt = item->AsStmt();
    if (stmt.subgraph_id() == -1) {
      for (auto& i : item->outlinks) {
        for (auto& j : i->outlinks) {
          if (j->IsStmt()) {
            auto& jstmt = j->AsStmt();
            if (jstmt.subgraph_id() == 0) inputvar = 1;
          }
        }
      }
      if (inputvar == 1) {
        for (auto& i : item->outlinks) i_nodes_[sub_id].insert(i);
      }
    }
    if (stmt.subgraph_id() != 0) continue;
    ChangeAllOutConnectedID(item, sub_id);
    sub_id++;
  }
  for (auto& i : nodes2rm_) {
    for (auto& item : i.second) {
      if (item->IsStmt()) {
        auto& stmt = item->AsStmt();
        LOG(INFO) << "nodes2rm_:" << stmt.op_type();
      } else if (item->IsArg()) {
        auto& arg = item->AsArg();
        LOG(INFO) << "nodes2rm_:" << arg.name;
      }
    }
  }
  for (auto& i : i_nodes_) {
    for (auto& item : i.second) {
      if (item->IsStmt()) {
        auto& stmt = item->AsStmt();
        LOG(INFO) << "i_nodes_: " << i.first << " " << stmt.op_type();
      } else if (item->IsArg()) {
        auto& arg = item->AsArg();
        LOG(INFO) << "i_nodes_: " << i.first << " " << arg.name;
      }
    }
  }
  for (auto& i : o_nodes_) {
    for (auto& item : i.second) {
      if (item->IsStmt()) {
        auto& stmt = item->AsStmt();
        LOG(INFO) << "o_nodes_:" << i.first << " " << stmt.op_type();
      } else if (item->IsArg()) {
        auto& arg = item->AsArg();
        LOG(INFO) << "o_nodes_: " << i.first << " " << arg.name;
      }
    }
  }
  return sub_id - 1;
}

int SubgraphProgramPass::FuseSubgraph(
    const std::unique_ptr<SSAGraph>& graph,
    const std::vector<std::string>& supported_op_types) {
  InitSubgraphID(graph, supported_op_types);
  nodes2rm_.clear();
  i_nodes_.clear();
  o_nodes_.clear();
  int num_subgraph = FuseSubgraphID(graph);
  LOG(INFO) << "detected " << num_subgraph << " subgraph";
  return num_subgraph;
}
}  // namespace subgraph
}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(subgraph_program_pass,
                  paddle::lite::mir::subgraph::SubgraphProgramPass);
