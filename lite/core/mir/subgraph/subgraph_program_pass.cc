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

std::unordered_map<int, std::unordered_set<Node*>>
SubgraphProgramPass::ClassifySubgraph(const std::unique_ptr<SSAGraph>& graph) {
  std::unordered_map<int, std::unordered_set<Node*>> op_nodes;
  for (auto& item : graph->StmtTopologicalOrder()) {
    if (!item->IsStmt()) continue;
    auto& stmt = item->AsStmt();
    int sub_id = stmt.subgraph_id();
    if (sub_id < 1) continue;
    if (!op_nodes.count(sub_id)) {
      op_nodes[sub_id] = std::unordered_set<Node*>();
    }
    op_nodes.at(sub_id).insert(item);
  }
  return op_nodes;
}

cpp::OpDesc SubgraphProgramPass::GenGraphOpDesc(
    const std::string& model_name,
    const std::vector<std::string>& in_var_names,
    const std::vector<std::string>& out_var_names) {
  cpp::OpDesc op_desc;
  op_desc.SetType("graph_op");
  op_desc.SetInput("Inputs", in_var_names);
  op_desc.SetOutput("Outputs", out_var_names);
  op_desc.SetAttr("model_name", model_name);
  return op_desc;
}

void SubgraphProgramPass::InsertNewNode(
    const std::unique_ptr<SSAGraph>& graph,
    const std::string& model_name,
    Scope* scope,
    const std::vector<Place>& valid_places,
    std::unordered_set<Node*> in_data_vars,
    std::unordered_set<Node*> in_wgt_vars,
    std::unordered_set<Node*> out_data_vars,
    std::unordered_set<Node*> out_unused_vars) {
  std::vector<std::string> in_var_names;
  std::vector<std::string> out_var_names;
  for (auto i : in_data_vars) {
    in_var_names.push_back(i->AsArg().name);
  }
  for (auto i : out_data_vars) {
    out_var_names.push_back(i->AsArg().name);
  }

  auto op_desc = GenGraphOpDesc(model_name, in_var_names, out_var_names);

  auto graph_op = LiteOpRegistry::Global().Create("graph_op");
  graph_op->Attach(op_desc, scope);
  auto* new_op_node = graph->GraphCreateInstructNode(graph_op, valid_places);

  for (auto& in_var : in_data_vars) {
    IR_NODE_LINK_TO(in_var, new_op_node);
  }
  for (auto& in_var : in_wgt_vars) {
    IR_NODE_LINK_TO(in_var, new_op_node);
  }
  for (auto& out_var : out_data_vars) {
    IR_OP_VAR_LINK(new_op_node, out_var);
  }
  for (auto& out_var : out_unused_vars) {
    IR_OP_VAR_LINK(new_op_node, out_var);
  }

  // assign context
  auto& inst = new_op_node->AsStmt();
  inst.picked_kernel().SetContext(
      ContextScheduler::Global().NewContext(inst.picked_kernel().target()));
}

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

void SubgraphProgramPass::FindInputOutputVars(
    const std::unordered_set<Node*>& op_nodes,
    std::unordered_set<Node*>* in_data_vars,
    std::unordered_set<Node*>* in_wgt_vars,
    std::unordered_set<Node*>* out_data_vars,
    std::unordered_set<Node*>* out_unused_vars) {
  for (auto& op_node : op_nodes) {
    for (auto& in_var : op_node->inlinks) {
      if (in_var->AsArg().is_weight) {
        in_wgt_vars->insert(in_var);
        continue;
      }
      if (!in_var->inlinks.empty()) {
        // var can only come from one op node, so use front
        auto* pre_op_node = in_var->inlinks.front();
        if (op_nodes.count(pre_op_node)) {
          continue;
        }
      }
      in_data_vars->insert(in_var);
    }
    for (auto& out_var : op_node->outlinks) {
      if (out_var->outlinks.empty()) {
        // the next op is empty so this var is actually unused
        out_unused_vars->insert(out_var);
        continue;
      }
      // var can have more than one next op node
      // so, if any one in the op_nodes then continue
      bool next_op_in_nodes = false;
      for (auto& next_op_node : out_var->outlinks) {
        if (op_nodes.count(next_op_node)) {
          next_op_in_nodes = true;
        }
      }
      if (next_op_in_nodes) {
        continue;
      }

      out_data_vars->insert(out_var);
    }
  }
}

std::unordered_set<const Node*> SubgraphProgramPass::GetNode2rm(
    const std::unordered_set<Node*>& op_nodes,
    const std::vector<std::unordered_set<Node*>>& excluded_nodes) {
  std::unordered_set<const Node*> nodes2rm(op_nodes.begin(), op_nodes.end());
  for (auto& op_node : op_nodes) {
    for (auto& in_var : op_node->inlinks) {
      if (!nodes2rm.count(in_var)) {
        nodes2rm.insert(in_var);
      }
    }
    for (auto& out_var : op_node->outlinks) {
      if (!nodes2rm.count(out_var)) {
        nodes2rm.insert(out_var);
      }
    }
  }
  // some nodes should not be removed
  for (auto& e : excluded_nodes) {
    for (auto& i : e) {
      if (nodes2rm.count(i)) {
        nodes2rm.erase(i);
      }
    }
  }
  return nodes2rm;
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
    for (auto& i : node->outlinks) {
      CHECK(i->IsStmt());
      auto& stmt = i->AsStmt();
      if (stmt.subgraph_id() == from_id) {
        stmt.SetSubgraphID(to_id);
        for (auto& o : i->outlinks) {
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
    }
    if (stmt.subgraph_id() != 0) continue;
    ChangeAllOutConnectedID(item, sub_id);
    sub_id++;
  }
  return sub_id - 1;
}

int SubgraphProgramPass::FuseSubgraph(
    const std::unique_ptr<SSAGraph>& graph,
    const std::vector<std::string>& supported_op_types) {
  InitSubgraphID(graph, supported_op_types);
  return FuseSubgraphID(graph);
}
}  // namespace subgraph
}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(subgraph_program_pass,
                  paddle::lite::mir::subgraph::SubgraphProgramPass);
