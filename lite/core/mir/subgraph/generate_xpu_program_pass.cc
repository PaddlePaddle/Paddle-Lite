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

std::shared_ptr<std::string> GenerateXPUProgramPass::CvtVarNode(
    lite::mir::Node* var_node, const Scope* scope) {
  CHECK(var_node->IsArg());
  const auto& arg = var_node->AsArg();
  VLOG(4) << "Convert var node " << arg.name;

  auto* var = scope->FindVar(arg.name);
  CHECK(var);
  auto* tensor = var->GetMutable<lite::Tensor>();
  CHECK(tensor);
  auto dims = tensor->dims();
  if (arg.is_weight) {
    auto wgt = std::make_shared<std::string>(arg.name);
    LOG(INFO) << "in convert const:" << arg.name;
    VLOG(4) << dims;
    // TODO(hong19860320)
    return wgt;
  } else {
    CHECK_EQ(dims.size(), 4);
    LOG(INFO) << "in convert data:" << arg.name;
    LOG(INFO) << dims;
    // TODO(hong19860320)
    auto data = std::make_shared<std::string>(arg.name);
    return data;
  }
  return nullptr;
}

void GenerateXPUProgramPass::CvtAllOpNodes(
    const std::vector<Node*>& nodes2cvt,
    lite::kernels::xpu::bridges::node_map_type* converted_vars) {
  const auto& bridges = lite::kernels::xpu::bridges::Factory::Instance();
  const auto& cvtfunc_map = bridges.AllFunctions();
  // return record all converted vars
  // op node's inputs must be found in converted_vars
  for (auto& node : nodes2cvt) {
    lite::kernels::xpu::bridges::node_map_type node_inputs;
    auto& stmt = node->AsStmt();
    for (auto& var_node : node->inlinks) {
      auto& arg = var_node->AsArg();
      // weight should be handled in the converter, so skip here
      if (arg.is_weight) {
        continue;
      }
      auto var_name = arg.name;
      if (!converted_vars->count(var_name)) {
        converted_vars->insert(
            std::make_pair(var_name, CvtVarNode(var_node, stmt.op()->scope())));
      }
      node_inputs.insert(*converted_vars->find(var_name));
    }
    auto node_outputs = cvtfunc_map.at(stmt.op_type())(stmt.op(), node_inputs);
    converted_vars->insert(node_outputs.begin(), node_outputs.end());
  }
}

std::string GenerateXPUProgramPass::BuildXPUGraph(
    const std::unordered_set<Node*>& op_nodes,
    const std::unordered_set<Node*>& in_data_vars,
    const std::unordered_set<Node*>& out_data_vars,
    int sub_id) {
  auto ordered_nodes = GetTopologicalOrder(op_nodes);
  lite::kernels::xpu::bridges::node_map_type converted_vars;
  CvtAllOpNodes(ordered_nodes, &converted_vars);

  std::vector<std::string> in_var_names;
  std::vector<std::string> out_var_names;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  for (auto i : in_data_vars) {
    auto argname = i->AsArg().name;
    in_var_names.push_back(argname);
    inputs.push_back(*converted_vars.at(argname));
  }
  for (auto i : out_data_vars) {
    auto argname = i->AsArg().name;
    out_var_names.push_back(argname);
    outputs.push_back(*converted_vars.at(argname));
  }

  std::string weight_var_name = "graph" + std::to_string(sub_id) + "_weights";
  auto any_op = (*op_nodes.begin())->AsStmt().op();
  auto weight = any_op->scope()->Var(weight_var_name)->GetMutable<Tensor>();
  weight->set_persistable(true);
  weight->set_precision(PRECISION(kInt8));
  // Compiling IR graph to XPU model and store mode data into weight tensor with
  // persistable=true, Sothat the model parser can recognize it and save it to
  // param files
  /*
  if (!lite::xpu::bridge::BuildModel(inputs, outputs, weight)) {
    LOG(WARNING) << "Build XPU failed subgraph " << sub_id;
    throw std::runtime_error("Build XPU failed subgraph.");
  }
  */
  LOG(INFO) << "[XPU] Build XPU Client success subgraph " << sub_id;
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
  LOG(INFO) << "Before NPU Pass \n" << Visualize(graph.get());
  const auto& bridges = lite::kernels::xpu::bridges::Factory::Instance();
  const auto& op_map = bridges.AllFunctions();
  std::vector<std::string> supported_op_types;
  for (auto& i : op_map) {
    LOG(INFO) << "Supported type: " << i.first;
    supported_op_types.push_back(i.first);
  }

  try {
    int num_subgraph = FuseSubgraph(graph, supported_op_types);
    InferOnce(graph);
    auto op_nodes_all = ClassifySubgraph(graph);
    CHECK_EQ(op_nodes_all.size(), num_subgraph);
    int id = 1;
    for (auto& op_nodes : op_nodes_all) {
      LOG(INFO) << "Converting subgraph_id:" << id;
      GenXPUSubgraph(graph, op_nodes.second, id);
      LOG(INFO) << "After XPU Pass Subgraph " << id << "\n"
                << Visualize(graph.get());
      id++;
    }
  } catch (...) {
    LOG(WARNING) << "Build XPU graph failed";
    throw std::runtime_error("Build XPU graph failed");
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
  LOG(INFO) << "insts.size " << insts_.size();
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
