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
void SubgraphProgramPass::ChangeAllOutConnectedID(Node* op_node,
                                                  int to_id,
                                                  int from_id) {
  if (!op_node) return;
  if (op_node->IsStmt()) {
    auto& stmt = op_node->AsStmt();
    if (stmt.subgraph_id() == from_id) {
      stmt.SetSubgraphID(to_id);
      for (auto& i : op_node->outlinks) {
        ChangeAllOutConnectedID(i, to_id, from_id);
      }
    } else {
      LOG(INFO) << "failed op type:" << stmt.op_type();
      return;
    }
  } else {
    for (auto& i : op_node->outlinks) {
      ChangeAllOutConnectedID(i, to_id, from_id);
    }
  }
}

int SubgraphProgramPass::FuseSubgraphID(
    const std::unique_ptr<SSAGraph>& graph) {
  int sub_id = 1;  // id start from 1 not 0
  for (auto& item : graph->StmtTopologicalOrder()) {
    // TODO(TJ): support node have vector inputs and output
    if (!item->IsStmt()) continue;
    auto& stmt = item->AsStmt();
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
