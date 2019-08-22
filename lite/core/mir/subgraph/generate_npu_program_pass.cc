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

#include "lite/core/mir/subgraph/generate_npu_program_pass.h"
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher.h"

#include "ai_ddk_lib/include/HiAiModelManagerService.h"
#include "ai_ddk_lib/include/graph/graph.h"
#include "ai_ddk_lib/include/graph/model.h"
#include "ai_ddk_lib/include/graph/op/all_ops.h"  // for ge::op::Data
#include "ai_ddk_lib/include/graph/operator_reg.h"
#include "lite/npu/bridge/paddle_use_npu_bridges.h"
#include "lite/npu/bridge/registry.h"
#include "lite/npu/bridge/utils.h"
#include "lite/npu/npu_helper.h"

namespace paddle {
namespace lite {
namespace mir {
namespace subgraph {

void GenerateNPUProgramPass::SubgraphSortHelper(
    Node* node,
    const std::unordered_set<Node*>& nodes_all,
    std::unordered_set<const Node*>* visited_nodes,
    std::vector<Node*>* ret) {
  for (auto& var_node : node->inlinks) {
    if (var_node->inlinks.empty()) continue;
    auto* op_node = var_node->inlinks.front();
    if (nodes_all.count(op_node) && !visited_nodes->count(op_node)) {
      SubgraphSortHelper(op_node, nodes_all, visited_nodes, ret);
    }
  }
  ret->push_back(node);
  visited_nodes->insert(node);
}

void GenerateNPUProgramPass::CvtOpNodes(
    const std::vector<Node*>& nodes2cvt,
    lite::npu::bridge::node_map_type* cvted_vars) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& cvtfunc_map = bridges.AllFunctions();
  // record all converted vars
  // op node's inputs must be found in cvted_vars
  for (auto& node : nodes2cvt) {
    lite::npu::bridge::node_map_type node_inputs;
    auto& stmt = node->AsStmt();
    for (auto& var_node : node->inlinks) {
      auto& arg = var_node->AsArg();
      if (arg.is_weight) continue;
      auto var_name = arg.name;
      if (!cvted_vars->count(var_name)) {
        cvted_vars->insert(std::make_pair(
            var_name,
            lite::npu::bridge::CvtNode(var_node, stmt.op()->scope())));
      }
      node_inputs.insert(*cvted_vars->find(var_name));
    }
    auto node_outputs = cvtfunc_map.at(stmt.op_type())(stmt.op(), node_inputs);
    cvted_vars->insert(node_outputs.begin(), node_outputs.end());
  }
}

void GenerateNPUProgramPass::GetIOVars(
    const std::vector<Node*>& nodes2cvt,
    const lite::npu::bridge::node_map_type& cvted_vars,
    std::unordered_set<const Node*>* nodes2rm,
    std::vector<Node*>* in_vars,
    std::vector<Node*>* out_vars,
    lite::npu::bridge::node_map_type* in_cvted_vars,
    lite::npu::bridge::node_map_type* out_cvted_vars) {
  std::unordered_set<Node*> op_nodes_all(nodes2cvt.begin(), nodes2cvt.end());
  for (auto& op_node : nodes2cvt) {
    for (auto& in_var : op_node->inlinks) {
      if (in_var->AsArg().is_weight) continue;
      auto* pre_op_node = in_var->inlinks.front();
      if (op_nodes_all.count(pre_op_node)) {
        nodes2rm->insert(in_var);
        continue;
      }
      in_vars->push_back(in_var);
      auto arg_name = in_var->AsArg().name;
      in_cvted_vars->insert(std::make_pair(arg_name, cvted_vars.at(arg_name)));
    }
    for (auto& out_var : op_node->outlinks) {
      if (out_var->outlinks.empty()) {
        nodes2rm->insert(out_var);
        continue;
      }
      auto* next_op_node = out_var->outlinks.front();

      if (op_nodes_all.count(next_op_node)) {
        nodes2rm->insert(out_var);
        continue;
      }
      out_vars->push_back(out_var);
      auto arg_name = out_var->AsArg().name;
      out_cvted_vars->insert(std::make_pair(arg_name, cvted_vars.at(arg_name)));
    }
  }
  nodes2rm->insert(nodes2cvt.begin(), nodes2cvt.end());
}

void GenerateNPUProgramPass::GenNPUGraphOpNode(
    const std::unique_ptr<SSAGraph>& graph,
    int sub_id,
    const std::unordered_set<Node*>& nodes_all) {
  std::unordered_set<const Node*> visited_nodes;
  std::vector<Node*> ret;
  for (auto& node : nodes_all) {
    if (!node->IsStmt()) continue;
    if (visited_nodes.count(node)) continue;
    SubgraphSortHelper(node, nodes_all, &visited_nodes, &ret);
  }

  lite::npu::bridge::node_map_type cvted_vars;
  CvtOpNodes(ret, &cvted_vars);

  std::unordered_set<const Node*> nodes2rm;
  std::vector<Node*> in_vars;
  std::vector<Node*> out_vars;
  lite::npu::bridge::node_map_type in_cvted_vars;
  lite::npu::bridge::node_map_type out_cvted_vars;
  GetIOVars(ret,
            cvted_vars,
            &nodes2rm,
            &in_vars,
            &out_vars,
            &in_cvted_vars,
            &out_cvted_vars);
  /*
  LOG(INFO)<<"----before remove: ";
  Visualize(graph.get());
  GraphSafeRemoveNodes(graph.get(), nodes2rm);
  LOG(INFO)<<"----after remove: ";
  Visualize(graph.get());
  */

  std::vector<std::string> in_vars_name;
  std::vector<std::string> out_vars_name;
  std::vector<ge::Operator> inputs;
  std::vector<ge::Operator> outputs;
  for (auto i : in_cvted_vars) {
    in_vars_name.push_back(i.first);
    inputs.push_back(*i.second);
  }
  for (auto i : out_cvted_vars) {
    out_vars_name.push_back(i.first);
    outputs.push_back(*i.second);
  }

  std::string model_name("hiai_npu_client_" + std::to_string(sub_id) + ".om");
  if (!npu::BuildNPUClient(inputs, outputs, model_name)) {
    LOG(FATAL) << "Build NPU failed subgraph " << sub_id;
  }
  LOG(INFO) << "[NPU] Build NPU Client success subgraph " << sub_id;

  cpp::OpDesc op_desc;
  op_desc.SetType("graph_op");
  std::vector<std::string> in_var_names;

  op_desc.SetInput("Inputs", in_vars_name);
  op_desc.SetOutput("Outputs", out_vars_name);
  op_desc.SetAttr("model_name", model_name);
  auto graph_op = LiteOpRegistry::Global().Create("graph_op");

  auto any_op = ret.front()->AsStmt().op();
  auto* scope = any_op->scope();
  graph_op->Attach(op_desc, scope);

  auto valid_places = any_op->valid_places();
  auto* new_op_node = graph->GraphCreateInstructNode(graph_op, valid_places);

  for (auto& in_var : in_vars) {
    IR_NODE_LINK_TO(in_var, new_op_node);
  }
  for (auto& out_var : out_vars) {
    IR_OP_VAR_LINK(new_op_node, out_var);
  }

  // assign context
  auto& inst = new_op_node->AsStmt();
  inst.picked_kernel().SetContext(
      ContextScheduler::Global().NewContext(inst.picked_kernel().target()));

  GraphSafeRemoveNodes(graph.get(), nodes2rm);
}

void GenerateNPUProgramPass::ConvertSubgraph(
    const std::unique_ptr<SSAGraph>& graph, int sub_num) {
  std::unordered_map<int, std::unordered_set<Node*>> nodes_all;
  int ops_num = 0;
  for (auto& item : graph->StmtTopologicalOrder()) {
    if (!item->IsStmt()) continue;
    ops_num++;
    auto& stmt = item->AsStmt();
    int sub_id = stmt.subgraph_id();
    if (sub_id < 1) continue;
    if (nodes_all.count(sub_id) == 0) {
      nodes_all[sub_id] = std::unordered_set<Node*>();
    }
    nodes_all.at(sub_id).insert(item);
  }

  for (int id = 1; id <= sub_num; ++id) {
    LOG(INFO) << "Converting subgraph_id:" << id;
    GenNPUGraphOpNode(graph, id, nodes_all.at(id));
  }
}

void GenerateNPUProgramPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  LOG(INFO) << "Before NPU Pass \n" << Visualize(graph.get());

  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& op_map = bridges.AllFunctions();
  std::vector<std::string> supported_op_types;
  for (auto& i : op_map) {
    LOG(INFO) << i.first;
    supported_op_types.push_back(i.first);
  }
  int num_subgraph = FuseSubgraph(graph, supported_op_types);
  LOG(INFO) << "detected " << num_subgraph << " NPU subgraph";

  InferOnce(graph);
  ConvertSubgraph(graph, num_subgraph);
  LOG(INFO) << "After NPU Pass \n" << Visualize(graph.get());

  for (auto& item : graph->StmtTopologicalOrder()) {
    if (item->IsStmt()) {
      auto& stmt = item->AsStmt();
      LOG(INFO) << stmt;
      insts_.emplace_back(stmt.op(), std::move(stmt.kernels().front()));
    }
  }
}

std::unique_ptr<RuntimeProgram> GenerateNPUProgramPass::GenProgram() {
  LOG(INFO) << "insts.size " << insts_.size();
  std::unique_ptr<RuntimeProgram> program(
      new RuntimeProgram(std::move(insts_)));
  return program;
}

}  // namespace subgraph
}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(generate_npu_program_pass,
                  paddle::lite::mir::subgraph::GenerateNPUProgramPass);
