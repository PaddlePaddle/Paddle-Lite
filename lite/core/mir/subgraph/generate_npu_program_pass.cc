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
#include "lite/npu/bridge/registry.h"
#include "lite/npu/npu_helper.h"

namespace paddle {
namespace lite {
namespace mir {

void GenerateNPUProgramPass::InferOnce(const std::unique_ptr<SSAGraph>& graph) {
  for (auto& item : graph->StmtTopologicalOrder()) {
    if (!item->IsStmt()) continue;
    auto& stmt = item->AsStmt();
    stmt.ClearSubgraphID();
    auto& op = stmt.op();
    op->CheckShape();
    op->InferShape();
    auto& kkks = stmt.kernels();
    if (!kkks.empty()) {
      auto& kk = stmt.kernels().front();
      if (kk) {
        kk->Launch();
      }
    }
  }
}

void GenerateNPUProgramPass::InitSubgraphID(
    const std::unique_ptr<SSAGraph>& graph) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  for (auto& item : graph->StmtTopologicalOrder()) {
    if (!item->IsStmt()) continue;
    auto& stmt = item->AsStmt();
    stmt.ClearSubgraphID();
    if (bridges.HasType(stmt.op_type())) {
      stmt.SetSubgraphID(0);
    }
  }
}

// mark current and all output supported nodes
void GenerateNPUProgramPass::ChangeAllOutConnectedID(Node* node,
                                                     int to_id,
                                                     int from_id) {
  if (!node) return;
  if (node->IsStmt()) {
    if (node->AsStmt().subgraph_id() == from_id) {
      node->AsStmt().SetSubgraphID(to_id);
    } else {
      return;
    }
  } else {
    for (auto& i : node->outlinks) {
      ChangeAllOutConnectedID(i, to_id, from_id);
    }
  }
}

int GenerateNPUProgramPass::FuseSubgraphID(
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

// call convert function from start node
// return if convert success and the nodes to remove
// return the output npu op
std::shared_ptr<ge::Operator> GenerateNPUProgramPass::CvtOpNodes(
    const lite::npu::bridge::Factory::map_type& cvtfunc_map,
    const Node* op_node,
    std::shared_ptr<ge::Operator> input,
    int sub_id,
    std::unordered_set<const Node*>* nodes2rm,
    key2nodes_t* matched) {
  if (!op_node->IsStmt()) {
    return nullptr;
  }
  auto* stmt = op_node->stmt();
  auto op_type = stmt->op_type();
  if (cvtfunc_map.find(op_type) == cvtfunc_map.end()) {
    return nullptr;
  }
  std::vector<std::shared_ptr<ge::Operator>> inputs{input};
  std::vector<std::shared_ptr<ge::Operator>> outputs =
      cvtfunc_map.at(op_type)(stmt->op(), inputs);
  auto output = outputs[0];
  if (!output) {
    return nullptr;
  }
  nodes2rm->insert(op_node);

  for (auto& var_node : op_node->outlinks) {
    for (auto& next_op_node : var_node->outlinks) {
      if (next_op_node->AsStmt().subgraph_id() != sub_id) {
        // this is the end condition
        // TODO(TJ): when enable more inputs and outputs this is bugy
        matched->insert(std::make_pair("Output", var_node));
        return output;
      } else {
        nodes2rm->insert(var_node);
        return CvtOpNodes(
            cvtfunc_map, next_op_node, output, sub_id, nodes2rm, matched);
      }
    }
  }
}

void GenerateNPUProgramPass::ConvertSubgraph(
    const std::unique_ptr<SSAGraph>& graph, int sub_num) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& cvtfunc_map = bridges.AllFunctions();
  std::unordered_set<const Node*> nodes2rm_all;

  auto items = graph->StmtTopologicalOrder();
  for (int id = 1; id <= sub_num; ++id) {
    for (auto& op_node : items) {
      std::unordered_set<const Node*> nodes2rm;
      if (!op_node->IsStmt()) continue;
      auto& stmt = op_node->AsStmt();
      if (stmt.subgraph_id() != id) continue;
      CHECK(bridges.HasType(stmt.op_type()));
      key2nodes_t matched;
      matched["target_op"] = op_node;
      auto& op = stmt.op();
      auto* scope = stmt.op()->scope();
      // prepare inputs data.
      std::string data_name = "data_subgraph_" + std::to_string(id);
      std::vector<std::shared_ptr<ge::op::Data>> npu_inputs;
      int name_id = 0;
      for (auto& arg_node : op_node->inlinks) {
        CHECK(arg_node->IsArg());
        auto& arg = arg_node->AsArg();
        if (!arg.is_weight) {
          auto* var = scope->FindVar(arg.name);
          CHECK(var);
          auto* tensor = var->GetMutable<lite::Tensor>();
          CHECK(tensor);
          auto dims = tensor->dims();
          CHECK_EQ(dims.size(), 4);
          // TODO(TJ): support more types and dims size
          ge::TensorDesc desc(ge::Shape(dims.Vectorize()),
                              ge::Format::FORMAT_NCHW,
                              ge::DataType::DT_FLOAT);
          auto data = std::make_shared<ge::op::Data>(data_name + "/" +
                                                     std::to_string(name_id));
          data->update_input_desc_x(desc);
          npu_inputs.push_back(data);
          matched["Input"] = arg_node;

          name_id++;
        }
      }
      // TODO(TJ): support more than 1
      CHECK_EQ(npu_inputs.size(), 1);
      auto npu_output =  // here is just one op yet, need to be vector
          CvtOpNodes(
              cvtfunc_map, op_node, npu_inputs[0], id, &nodes2rm, &matched);
      if (npu_output) {
        ge::Graph npu_subgraph("npu_subgraph_" + id);
        std::vector<ge::Operator> inputs{*npu_inputs[0]};
        std::vector<ge::Operator> outputs{*npu_output};
        std::string model_name("_npu_client_" + std::to_string(id) + "_");
        if (!npu::BuildNPUClient(inputs, outputs, model_name)) {
          // build failed, so this subgraph is abandoned
          nodes2rm.clear();
          continue;
        }

        // Then  InsertNewNode(graph, matched); make one function
        // use fuser?
        cpp::OpDesc op_desc = *matched.at("target_op")->stmt()->op_info();
        op_desc.SetType("graph_op");
        op_desc.SetInput("Input", {matched.at("Input")->arg()->name});
        op_desc.SetOutput("Out", {matched.at("Output")->arg()->name});
        op_desc.SetAttr("model_name", model_name);
        auto graph_op = LiteOpRegistry::Global().Create("graph_op");
        auto target_op = matched.at("target_op")->stmt()->op();
        auto* scope = target_op->scope();
        graph_op->Attach(op_desc, scope);
        auto* new_op_node =
            graph->GraphCreateInstructNode(graph_op, target_op->valid_places());
        IR_NODE_LINK_TO(matched.at("Input"), new_op_node);
        IR_NODE_LINK_TO(new_op_node, matched.at("Output"));

        // TODO(TJ): add context and kernels
        //           GraphCreateInstructNode

        // // TODO(Superjomn) remove one valid_places here.
        // op->SetValidPlaces(valid_places);
        // auto &new_node = node_storage_.back();
        // auto kernels = op->CreateKernels(valid_places);
        // node_storage_.back().AsStmt(op->op_type_, std::move(kernels), op);

        // CHECK(new_node.inlinks.empty()) << "duplicate Build found";
        // CHECK(new_node.outlinks.empty()) << "duplicate Build found";
        // return &node_storage_.back(); */
        //   // stmt.kernels().clear();
        //   // stmt.kernels().emplace_back(std::move(scored.front().second));

      } else {
        nodes2rm.clear();
      }
      if (!nodes2rm.empty()) {
        nodes2rm_all.insert(nodes2rm.begin(), nodes2rm.end());
      }
    }
  }
  // remove all unused node once
  GraphSafeRemoveNodes(graph.get(), nodes2rm_all);
}

void GenerateNPUProgramPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  VLOG(4) << "final program \n" << Visualize(graph.get());

  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& op_map = bridges.AllFunctions();
  InferOnce(graph);
  InitSubgraphID(graph);
  int num_subgraph = FuseSubgraphID(graph);
  ConvertSubgraph(graph, num_subgraph);
  // auto graph1 = GenerateFusedGraph(std::move(graph));
  // GraphSafeRemoveNodes(graph, nodes2rm);

  for (auto& item : graph->StmtTopologicalOrder()) {
    if (item->IsStmt()) {
      auto& stmt = item->AsStmt();
      LOG(INFO) << stmt;
      auto* scope = stmt.op()->scope();
      auto* op_info = stmt.op()->op_info();
      auto& op = stmt.op();

      LOG(INFO) << "op type: " << op_info->Type();
      if (op_info->Type() != "fetch") {
        if (op_info->output_argnames().size() >= 1) {
          auto first_output_name = op_info->output_argnames().front();
          LOG(INFO) << ",first_output_name: " << first_output_name;
          if (op_info->Output(first_output_name).size() >= 1) {
            auto var_name = op_info->Output(first_output_name).front();
            LOG(INFO) << ",fisrt output var_name:" << var_name;

            auto* var = scope->FindVar(var_name);
            if (var) {
              auto* tensor = var->GetMutable<lite::Tensor>();
              LOG(INFO) << ",tensor :" << tensor;

              if (tensor && op_info->Type() != "feed") {
                LOG(INFO) << ", dims:" << tensor->dims();
              }
            }
          }
        }
      }
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

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(generate_npu_program_pass,
                  paddle::lite::mir::GenerateNPUProgramPass);
