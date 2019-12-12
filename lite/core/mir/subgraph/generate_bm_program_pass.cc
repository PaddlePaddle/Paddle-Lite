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

#include "lite/core/mir/subgraph/generate_bm_program_pass.h"
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher.h"

#include "lite/kernels/bm/bridges/paddle_use_bm_bridges.h"
#include "lite/kernels/bm/bridges/registry.h"
#include "bmcompiler_if.h"
#include "bmlog.hpp"

namespace paddle {
namespace lite {
namespace mir {
namespace subgraph {

std::shared_ptr<void*> GenerateBMProgramPass::CvtVarNode(
    lite::mir::Node* var_node, const Scope* scope) {
  return nullptr;
}

void GenerateBMProgramPass::CvtAllOpNodes(
    const std::vector<Node*>& nodes2cvt,
    lite::kernels::bm::bridges::node_map_type* converted_vars) {
    
    const auto& bridges = lite::kernels::bm::bridges::Factory::Instance();
    const auto& cvtfunc_map = bridges.AllFunctions();
    
    lite::kernels::bm::bridges::graph_ctx_type ctx;
    ctx.bm_compiler_handle = create_bmcompiler("BM1684");
    CHECK(ctx.bm_compiler_handle != nullptr);

    //bmlog::init("paddle_bitmain");
    //bmlog::set_v(3);

    for (auto& node : nodes2cvt) {
        lite::kernels::bm::bridges::node_map_type node_inputs;
        auto& stmt = node->AsStmt();
        
        for (auto& var_node : node->inlinks) {
            auto& arg = var_node->AsArg();
            // weight should be handled in the converter, so skip here
            if (arg.is_weight) {
                continue;
            }
            auto var_name = arg.name;
            if (!converted_vars->count(var_name)) {
                converted_vars->insert(std::make_pair(var_name, var_name));
            }
            node_inputs.insert(*converted_vars->find(var_name));
        }

        auto node_outputs = cvtfunc_map.at(stmt.op_type())(stmt.op(), &ctx, node_inputs);
        converted_vars->insert(node_outputs.begin(), node_outputs.end());
    }
    
    std::string net_name = "paddle_bitmain";
     __bmcompile_opt(ctx.bm_compiler_handle, const_cast<char*>(net_name.c_str()), 2);
    finish_bmcompiler(ctx.bm_compiler_handle);
}

void GenerateBMProgramPass::GenSubgraph(
    const std::unique_ptr<SSAGraph>& graph,
    const std::unordered_set<Node*>& op_nodes,
    int sub_id) {

  std::unordered_set<Node*> in_data_vars;
  std::unordered_set<Node*> in_wgt_vars;
  std::unordered_set<Node*> out_data_vars;
  std::unordered_set<Node*> out_unused_vars;
  FindInputOutputVars(
      op_nodes, &in_data_vars, &in_wgt_vars, &out_data_vars, &out_unused_vars);

  auto ordered_nodes = GetTopologicalOrder(op_nodes);
  lite::kernels::bm::bridges::node_map_type converted_vars;
  CvtAllOpNodes(ordered_nodes, &converted_vars);
}

void GenerateBMProgramPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  const auto& bridges = lite::kernels::bm::bridges::Factory::Instance();
  const auto& op_map = bridges.AllFunctions();
  std::vector<std::string> supported_op_types;
  for (auto& i : op_map) {
    //LOG(INFO) << "[BM] Supported type: " << i.first;
    supported_op_types.push_back(i.first);
  }
  
  int num_subgraph = FuseSubgraph(graph, supported_op_types);
  InferOnce(graph);
  auto op_nodes_all = ClassifySubgraph(graph);
  CHECK_EQ(op_nodes_all.size(), num_subgraph);

  int id = 1;
  for (auto& op_nodes : op_nodes_all) {
    //LOG(INFO) << "[BM] Converting Subgraph " << id;
    GenSubgraph(graph, op_nodes.second, id);
    id++;
  }
  
}

std::unique_ptr<RuntimeProgram> GenerateBMProgramPass::GenProgram() {
  std::unique_ptr<RuntimeProgram> program(
      new RuntimeProgram(std::move(insts_)));
  return program;
}

}  // namespace subgraph
}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(generate_bm_program_pass,
                  paddle::lite::mir::subgraph::GenerateBMProgramPass)
    .BindTargets({TARGET(kBM)});
