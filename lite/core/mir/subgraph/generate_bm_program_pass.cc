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

namespace paddle {
namespace lite {
namespace mir {
namespace subgraph {

std::shared_ptr<ge::Operator> GenerateBMProgramPass::CvtVarNode(
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
    auto wgt = std::make_shared<ge::op::Const>(arg.name);
    LOG(INFO) << " Convert const var node " << arg.name;
    VLOG(4) << dims;
    wgt->set_attr_value(lite::npu::CvtTensor(tensor));
    return wgt;
  } else {
    CHECK_EQ(dims.size(), 4);
    LOG(INFO) << "[NPU] Convert data var node " << arg.name;
    LOG(INFO) << dims;
    // TODO(xxx): support more types and dims size
    ge::TensorDesc desc(ge::Shape(dims.Vectorize()),
                        ge::Format::FORMAT_NCHW,
                        ge::DataType::DT_FLOAT);

    //   auto size = desc.GetShape().GetShapeSize();
    //  ge::TensorUtils::SetSize(desc, size*sizeof(float));
    //  ge::TensorUtils::SetRealDimCnt(desc, 4);
    auto data = std::make_shared<ge::op::Data>(arg.name);
    data->update_input_desc_x(desc);
    return data;
  }
  return nullptr;
}

void GenerateNPUProgramPass::CvtAllOpNodes(
    const std::vector<Node*>& nodes2cvt,
    lite::kernels::npu::bridges::node_map_type* converted_vars) {
  const auto& bridges = lite::kernels::npu::bridges::Factory::Instance();
  const auto& cvtfunc_map = bridges.AllFunctions();
  // return record all converted vars
  // op node's inputs must be found in converted_vars
  for (auto& node : nodes2cvt) {
    lite::kernels::npu::bridges::node_map_type node_inputs;
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

std::string GenerateNPUProgramPass::BuildNPUGraph(
    const std::unordered_set<Node*>& op_nodes,
    const std::unordered_set<Node*>& in_data_vars,
    const std::unordered_set<Node*>& out_data_vars,
    int sub_id) {
  auto ordered_nodes = GetTopologicalOrder(op_nodes);
  lite::kernels::npu::bridges::node_map_type converted_vars;
  CvtAllOpNodes(ordered_nodes, &converted_vars);

  std::vector<std::string> in_var_names;
  std::vector<std::string> out_var_names;
  std::vector<ge::Operator> inputs;
  std::vector<ge::Operator> outputs;
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
  // Compiling IR graph to NPU model and store mode data into weight tensor with
  // persistable=true, Sothat the model parser can recognize it and save it to
  // param files
  if (!lite::npu::BuildModel(inputs, outputs, weight)) {
    LOG(WARNING) << "[NPU] Build NPU graph failed (subgraph=" << sub_id << ")";
    throw std::runtime_error("Build NPU graph failed.");
  }
  LOG(INFO) << "[NPU] Build NPU graph success (subgraph=" << sub_id << ")";
  return weight_var_name;
}

void GenerateBMProgramPass::GenSubgraph(
    const std::unique_ptr<SSAGraph>& graph,
    const std::unordered_set<Node*>& op_nodes,
    int sub_id) {
#if 0
  std::unordered_set<Node*> in_data_vars;
  std::unordered_set<Node*> in_wgt_vars;
  std::unordered_set<Node*> out_data_vars;
  std::unordered_set<Node*> out_unused_vars;
  FindInputOutputVars(
      op_nodes, &in_data_vars, &in_wgt_vars, &out_data_vars, &out_unused_vars);

  auto weight_var_name =
      BuildNPUGraph(op_nodes, in_data_vars, out_data_vars, sub_id);

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
#endif
}

void GenerateBMProgramPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  
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
