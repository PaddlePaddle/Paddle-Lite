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

#include "lite/core/mir/subgraph/generate_xpu_program_pass.h"
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher.h"

#include "lite/backends/xpu/builder.h"
#include "lite/kernels/xpu/bridges/paddle_use_xpu_bridges.h"
#include "lite/kernels/xpu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace mir {
namespace subgraph {

std::shared_ptr<xtcl::xExpr> GenerateXPUProgramPass::CvtVarNode(
    lite::kernels::xpu::bridges::graph_ctx_type* graph_ctx,
    lite::mir::Node* var_node,
    const Scope* scope) {
  CHECK(var_node->IsArg());
  const auto& arg = var_node->AsArg();
  auto var_name = arg.name;
  VLOG(4) << "[XPU] Convert var node " << var_name;

  auto* var = scope->FindVar(var_name);
  CHECK(var);
  auto* tensor = var->GetMutable<lite::Tensor>();
  CHECK(tensor);
  auto dims = tensor->dims();
  auto cvted_var_node =
      std::make_shared<xtcl::xExpr>(graph_ctx->builder->CreateTensor(
          var_name, lite::xpu::CvtShape(dims), ::xtcl::Float(32)));
  if (arg.is_weight) {
    auto cvted_var_tensor = lite::xpu::CvtTensor(tensor);
    graph_ctx->params->emplace(std::make_pair(var_name, *cvted_var_tensor));
  }
  return cvted_var_node;
}

void GenerateXPUProgramPass::CvtAllOpNodes(
    const std::vector<Node*>& op_nodes,
    lite::kernels::xpu::bridges::graph_ctx_type* graph_ctx,
    lite::kernels::xpu::bridges::node_map_type* cvted_var_nodes) {
  const auto& bridges = lite::kernels::xpu::bridges::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  // return record all converted vars
  // op node's inputs must be found in converted_vars
  for (auto& node : op_nodes) {
    lite::kernels::xpu::bridges::node_map_type input_nodes;
    auto& stmt = node->AsStmt();
    for (auto& var_node : node->inlinks) {
      auto& arg = var_node->AsArg();
      // weight should be handled in the converter, so skip here
      if (arg.is_weight) {
        continue;
      }
      auto var_name = arg.name;
      if (!cvted_var_nodes->count(var_name)) {
        cvted_var_nodes->insert(std::make_pair(
            var_name, CvtVarNode(graph_ctx, var_node, stmt.op()->scope())));
      }
      input_nodes.insert(*cvted_var_nodes->find(var_name));
    }
    auto output_nodes =
        supported_lists.at(stmt.op_type())(stmt.op(), graph_ctx, input_nodes);
    cvted_var_nodes->insert(output_nodes.begin(), output_nodes.end());
  }
}

std::string GenerateXPUProgramPass::BuildXPUGraph(
    const std::unordered_set<Node*>& op_nodes,
    const std::unordered_set<Node*>& in_data_vars,
    const std::unordered_set<Node*>& out_data_vars,
    int sub_id) {
  auto ordered_op_nodes = GetTopologicalOrder(op_nodes);
  lite::kernels::xpu::bridges::graph_ctx_type graph_ctx;
  graph_ctx.builder = std::make_shared<xtcl::network::xNetworkBuilder>();
  graph_ctx.params =
      std::make_shared<xtcl::network::xTensorCompiler::ParamNDArrayMap>();
  lite::kernels::xpu::bridges::node_map_type cvted_var_nodes;
  CvtAllOpNodes(ordered_op_nodes, &graph_ctx, &cvted_var_nodes);

  std::string weight_var_name = "graph" + std::to_string(sub_id) + "_weights";
  auto any_op = (*op_nodes.begin())->AsStmt().op();
  auto weight = any_op->scope()->Var(weight_var_name)->GetMutable<Tensor>();
  weight->set_persistable(true);
  weight->set_precision(PRECISION(kInt8));
  // Compiling graph to XPU model and store mode data into weight tensor with
  // persistable=true, Sothat the model parser can recognize it and save it to
  // param files
  std::vector<std::shared_ptr<xtcl::xExpr>> ordered_cvted_var_nodes;
  for (auto out_data_var : out_data_vars) {
    auto var_name = out_data_var->AsArg().name;
    ordered_cvted_var_nodes.push_back(cvted_var_nodes[var_name]);
  }
  if (!lite::xpu::BuildModel(graph_ctx.builder,
                             graph_ctx.params,
                             &ordered_cvted_var_nodes,
                             weight)) {
    LOG(WARNING) << "[XPU] Build XPU graph failed (subgraph=" << sub_id << ")";
    throw std::runtime_error("[XPU] Build XPU graph failed.");
  }
  LOG(INFO) << "[XPU] Build XPU graph success (subgraph=" << sub_id << ")";
  return weight_var_name;
}

void GenerateXPUProgramPass::GenXPUSubgraph(
    const std::unique_ptr<SSAGraph>& graph,
    const std::unordered_set<Node*>& op_nodes,
    int sub_id) {
  std::unordered_set<Node*> in_data_vars;
  std::unordered_set<Node*> in_wgt_vars;
  std::unordered_set<Node*> out_data_vars;
  std::unordered_set<Node*> out_unused_vars;
  FindInputOutputVars(
      op_nodes, &in_data_vars, &in_wgt_vars, &out_data_vars, &out_unused_vars);

  auto weight_var_name =
      BuildXPUGraph(op_nodes, in_data_vars, out_data_vars, sub_id);

  auto any_op = (*op_nodes.begin())->AsStmt().op();
  InsertNewNode(graph,
                weight_var_name,
                any_op->scope(),
                any_op->valid_places(),
                in_data_vars,
                in_wgt_vars,
                out_data_vars,
                out_unused_vars);

  auto nodes2rm = GetNode2rm(
      op_nodes, {in_data_vars, in_wgt_vars, out_data_vars, out_unused_vars});

  GraphSafeRemoveNodes(graph.get(), nodes2rm);
}

void GenerateXPUProgramPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  LOG(INFO) << "[XPU] Before XPU Pass \n" << Visualize(graph.get());
  const auto& bridges = lite::kernels::xpu::bridges::Factory::Instance();
  const auto& op_map = bridges.AllFunctions();
  std::vector<std::string> supported_op_types;
  for (auto& i : op_map) {
    LOG(INFO) << "[XPU] Supported type: " << i.first;
    supported_op_types.push_back(i.first);
  }

  try {
    int num_subgraph = FuseSubgraph(graph, supported_op_types);
    InferOnce(graph);
    auto op_nodes_all = ClassifySubgraph(graph);
    CHECK_EQ(op_nodes_all.size(), num_subgraph);
    int id = 1;
    for (auto& op_nodes : op_nodes_all) {
      LOG(INFO) << "[XPU] Converting Subgraph " << id;
      GenXPUSubgraph(graph, op_nodes.second, id);
      LOG(INFO) << "[XPU] After XPU Pass Subgraph " << id << "\n"
                << Visualize(graph.get());
      id++;
    }
  } catch (...) {
    LOG(WARNING) << "[XPU] Build XPU graph failed.";
    throw std::runtime_error("[XPU] Build XPU graph failed.");
  }

  for (auto& item : graph->StmtTopologicalOrder()) {
    if (item->IsStmt()) {
      auto& stmt = item->AsStmt();
      LOG(INFO) << stmt;
      insts_.emplace_back(stmt.op(), std::move(stmt.kernels().front()));
    }
  }
}

std::unique_ptr<RuntimeProgram> GenerateXPUProgramPass::GenProgram() {
  LOG(INFO) << "[XPU] program insts.size=" << insts_.size();
  std::unique_ptr<RuntimeProgram> program(
      new RuntimeProgram(std::move(insts_)));
  return program;
}

}  // namespace subgraph
}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(generate_xpu_program_pass,
                  paddle::lite::mir::subgraph::GenerateXPUProgramPass)
    .BindTargets({TARGET(kXPU)});
