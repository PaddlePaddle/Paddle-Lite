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
#include "lite/npu/bridge/paddle_use_npu_bridges.h"
#include "lite/npu/bridge/registry.h"
#include "lite/npu/bridge/utils.h"
#include "lite/npu/npu_helper.h"

namespace paddle {
namespace lite {
namespace mir {
namespace subgraph {

// call convert function from start node
// return if convert success and the nodes to remove
// return the output npu op
lite::npu::bridge::node_map_type GenerateNPUProgramPass::CvtOpNodes(
    const lite::npu::bridge::cvt_map_type& cvtfunc_map,
    const Node* op_node,
    const lite::npu::bridge::node_map_type& inputs_map,
    int sub_id,
    std::unordered_set<const Node*>* nodes2rm,
    key2nodes_t* matched) {
  lite::npu::bridge::node_map_type failed;
  if (!op_node->IsStmt()) {
    LOG(INFO) << "stop return failed";
    return failed;
  }
  auto* stmt = op_node->stmt();
  auto op_type = stmt->op_type();
  LOG(INFO) << "cvt op type: " << op_type;

  if (stmt->subgraph_id() != sub_id) {
    LOG(INFO) << "return as subgraph_id(" << stmt->subgraph_id()
              << ") != sub_id(" << sub_id << ")";
    return failed;
  } else {
    CHECK(cvtfunc_map.count(op_type)) << "Should be supported " << op_type
                                      << ", with subgraph_id: " << sub_id;
  }

  auto outputs_map = cvtfunc_map.at(op_type)(stmt->op(), inputs_map);
  if (outputs_map.empty()) {
    return outputs_map;
  }

  nodes2rm->insert(op_node);
  for (auto& var_node : op_node->outlinks) {
    for (auto& next_op_node : var_node->outlinks) {
      LOG(INFO) << "next op type: " << next_op_node->AsStmt().op_type();
      if (next_op_node->AsStmt().subgraph_id() != sub_id) {
        // this is the end condition
        // TODO(TJ): when enable more inputs and outputs this is bugy
        LOG(INFO) << "--- should return once ---";
        // TODO(TJ): matched output could be vector
        matched->insert(std::make_pair("Output", var_node));
        return outputs_map;
      } else {
        // LOG(INFO) << "argnames: ";
        // for (auto sss : next_op_node->AsStmt().op_info()->input_argnames()) {
        //   LOG(INFO) << sss;
        // }
        // LOG(INFO) << "input argnames: ";
        // for (auto sss : next_op_node->AsStmt().op_info()->input_names()) {
        //   LOG(INFO) << sss;
        // }
        for (auto& i_node : next_op_node->inlinks) {
          CHECK(i_node->IsArg());
          auto& arg = i_node->AsArg();
          LOG(INFO) << arg.name;
          if (outputs_map.count(arg.name)) continue;
          if (!arg.is_weight) {
            LOG(INFO) << "Data arg name:" << arg.name;
            outputs_map.insert(std::make_pair(
                arg.name,
                lite::npu::bridge::CvtNode(
                    i_node, next_op_node->AsStmt().op()->scope())));
          }
        }
        nodes2rm->insert(var_node);
        return CvtOpNodes(
            cvtfunc_map, next_op_node, outputs_map, sub_id, nodes2rm, matched);
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
    LOG(INFO) << "Converting subgraph_id:" << id;
    for (auto& op_node : items) {
      std::unordered_set<const Node*> nodes2rm;
      if (!op_node->IsStmt()) continue;
      auto& stmt = op_node->AsStmt();
      if (stmt.subgraph_id() != id) continue;
      CHECK(bridges.HasType(stmt.op_type()));
      key2nodes_t matched;
      matched["target_op"] = op_node;
      auto& op = stmt.op();
      auto* scope = op->scope();
      // prepare inputs data.
      std::string data_name = "data_subgraph_" + std::to_string(id);
      lite::npu::bridge::node_map_type npu_inputs_map;
      int name_id = 0;
      LOG(INFO) << "op_type: " << stmt.op_type();
      std::vector<std::string> actual_input_argnames;
      for (auto& arg_node : op_node->inlinks) {
        CHECK(arg_node->IsArg());
        const auto& arg = arg_node->AsArg();
        if (!arg_node->AsArg().is_weight) {
          LOG(INFO) << "Input arg name: " << arg.name;
          npu_inputs_map.insert(std::make_pair(
              arg.name, lite::npu::bridge::CvtNode(arg_node, scope)));
          // TODO(TJ): Here matched inputs should also be input vector
          matched["Input"] = arg_node;
          name_id++;
        }
      }
      CHECK_EQ(name_id, 1) << "mobilenetv1 only have one input data!";
      auto npu_outputs_map = CvtOpNodes(
          cvtfunc_map, op_node, npu_inputs_map, id, &nodes2rm, &matched);
      if (!npu_outputs_map.empty()) {
        LOG(INFO) << "[NPU] subgraph " << id << ": output not empty ";
        std::vector<ge::Operator> inputs;
        std::vector<ge::Operator> outputs;
        for (auto& i : npu_inputs_map) {
          LOG(INFO) << "input data argname:" << i.first
                    << ", ptr: " << i.second;
          inputs.emplace_back(*(i.second));
        }
        for (auto& i : npu_outputs_map) {
          LOG(INFO) << "output data argname:" << i.first
                    << ", ptr: " << i.second;
          outputs.emplace_back(*(i.second));
        }

        std::string model_name("hiai_npu_client_" + std::to_string(id) + ".om");
        if (!npu::BuildNPUClient(inputs, outputs, model_name)) {
          // build failed, so this subgraph is abandoned
          nodes2rm.clear();
          LOG(WARNING) << "Build NPU failed subgraph " << id;
          break;
        }
        LOG(INFO) << "[NPU] Build NPU Client success subgraph " << id;

        // Then InsertNewNode(graph, matched); make one function
        cpp::OpDesc op_desc = *matched.at("target_op")->stmt()->op_info();
        op_desc.SetType("graph_op");
        // change to vectors
        op_desc.SetInput("Inputs", {matched.at("Input")->arg()->name});
        op_desc.SetOutput("Outputs", {matched.at("Output")->arg()->name});
        op_desc.SetAttr("model_name", model_name);
        auto graph_op = LiteOpRegistry::Global().Create("graph_op");
        auto target_op = matched.at("target_op")->stmt()->op();
        auto* scope = target_op->scope();
        CHECK(scope);
        CHECK(graph_op);
        graph_op->Attach(op_desc, scope);

        auto valid_places =
            target_op->valid_places();  // TODO(TJ): add npu place?
        auto* new_op_node =
            graph->GraphCreateInstructNode(graph_op, valid_places);

        IR_NODE_LINK_TO(matched.at("Input"), new_op_node);
        IR_NODE_LINK_TO(new_op_node, matched.at("Output"));

        // assign context
        auto& inst = new_op_node->AsStmt();
        inst.picked_kernel().SetContext(ContextScheduler::Global().NewContext(
            inst.picked_kernel().target()));

        if (!nodes2rm.empty()) {
          nodes2rm_all.insert(nodes2rm.begin(), nodes2rm.end());
        }
        break;
      }  // if npu output success
    }    // for op_nodes
  }      // for subgraph id
  // remove all unused node once
  GraphSafeRemoveNodes(graph.get(), nodes2rm_all);
  // clear all npu ops
  npu::OpList::Global().clear();
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
  // auto graph1 = GenerateFusedGraph(std::move(graph));
  // GraphSafeRemoveNodes(graph, nodes2rm);
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

// USE_LITE_OP(graph_op);
