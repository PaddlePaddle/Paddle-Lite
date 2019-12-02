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

#include "lite/core/mir/subgraph/subgraph_pass.h"
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher.h"
#include "lite/core/mir/subgraph/subgraph_detector.h"
#include "lite/operators/subgraph_op.h"

namespace paddle {
namespace lite {
namespace mir {

void SubgraphPass::InsertNewNode(SSAGraph* graph,
                                 int subgraph_idx,
                                 const std::vector<Node*>& subgraph_nodes) {
  // Create and attach a new subgraph op
  cpp::OpDesc subgraph_op_desc;
  subgraph_op_desc.SetType("subgraph");

  // Create a new sub block desc for storing all of Ops an Vars of the target
  // subgraph and sub_block_idx is set as a attribute of subgraph op,
  // sub_block_idx < 0 means it's a new subgraph op
  int sub_block_idx = -(subgraph_idx + 1);
  auto sub_block_desc = new cpp::BlockDesc();
  sub_block_desc->ClearOps();
  sub_block_desc->ClearVars();
  for (auto& op_node : subgraph_nodes) {
    auto sub_block_op_desc = sub_block_desc->AddOp<cpp::OpDesc>();
    *sub_block_op_desc = *op_node->AsStmt().op_info();
    sub_block_op_desc->SetAttr(
        kKernelTypeAttr,
        op_node->AsStmt().picked_kernel().SerializedKernelType());
  }
  subgraph_op_desc.SetAttr<int32_t>("sub_block", sub_block_idx);

  // Extract input and output nodes from the target subgraph
  std::unordered_set<Node*> input_var_nodes;
  std::unordered_set<Node*> weight_var_nodes;
  std::unordered_set<Node*> output_var_nodes;
  std::unordered_set<Node*> local_var_nodes;
  std::unordered_set<Node*> unused_var_nodes;
  ExtractInputsOutputs(subgraph_nodes,
                       &input_var_nodes,
                       &weight_var_nodes,
                       &output_var_nodes,
                       &local_var_nodes,
                       &unused_var_nodes);

  // Set input and output name mapping which stores the real inputs and
  // outputs
  std::vector<std::string> input_var_names;
  std::vector<std::string> output_var_names;
  for (auto& var_node : input_var_nodes) {
    input_var_names.push_back(var_node->AsArg().name);
  }
  for (auto& var_node : output_var_nodes) {
    output_var_names.push_back(var_node->AsArg().name);
  }
  subgraph_op_desc.SetAttr<std::vector<std::string>>("input_name_mapping",
                                                     input_var_names);
  subgraph_op_desc.SetAttr<std::vector<std::string>>("output_name_mapping",
                                                     output_var_names);

  // Set all of the inputs and outputs to the target subgraph op
  // To prevent vars are removed in RuntimeProgram::UpdateVarsOfProgram()
  for (auto& var_node : weight_var_nodes) {
    input_var_names.push_back(var_node->AsArg().name);
  }
  for (auto& var_node : local_var_nodes) {
    output_var_names.push_back(var_node->AsArg().name);
  }
  for (auto& var_node : unused_var_nodes) {
    output_var_names.push_back(var_node->AsArg().name);
  }
  subgraph_op_desc.SetInput("Inputs", input_var_names);
  subgraph_op_desc.SetOutput("Outputs", output_var_names);
  auto subgraph_op = LiteOpRegistry::Global().Create("subgraph");
  static_cast<operators::SubgraphOp*>(subgraph_op.get())
      ->SetSubBlock(sub_block_desc);
  auto any_op = (*subgraph_nodes.begin())->AsStmt().op();
  subgraph_op->Attach(subgraph_op_desc, any_op->scope());

  // Create and add a new subgraph node into the graph
  auto subgraph_op_node =
      graph->GraphCreateInstructNode(subgraph_op, any_op->valid_places());
  for (auto& var_node : input_var_nodes) {
    IR_NODE_LINK_TO(var_node, subgraph_op_node);
  }
  for (auto& var_node : weight_var_nodes) {
    IR_NODE_LINK_TO(var_node, subgraph_op_node);
  }
  for (auto& var_node : output_var_nodes) {
    IR_OP_VAR_LINK(subgraph_op_node, var_node);
  }
  for (auto& var_node : local_var_nodes) {
    IR_OP_VAR_LINK(subgraph_op_node, var_node);
  }
  for (auto& var_node : unused_var_nodes) {
    IR_OP_VAR_LINK(subgraph_op_node, var_node);
  }

  // Create and assign the context to the picked kernel of the new subgraph
  // node
  auto& inst = subgraph_op_node->AsStmt();
  inst.picked_kernel().SetContext(
      ContextScheduler::Global().NewContext(inst.picked_kernel().target()));

  // Remove subgraph nodes and unused var nodes
  auto nodes2rm = GetNodes2RM(subgraph_nodes,
                              {input_var_nodes,
                               weight_var_nodes,
                               output_var_nodes,
                               local_var_nodes,
                               unused_var_nodes});
  GraphSafeRemoveNodes(graph, nodes2rm);
}

void SubgraphPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  auto teller = [](Node* node) {
    if (!node->IsStmt()) return false;
    auto& stmt = node->AsStmt();
    auto op_type = stmt.op_type();
    // std::vector<std::string> supported_or_unsupported_op_types = {
    //    "feed", "fetch", "subgraph"};
    // bool is_supported_op_types = false;
    std::vector<std::string> supported_or_unsupported_op_types = {
        "elementwise_sub", "elementwise_mul", "pool2d"};
    bool is_supported_op_types = true;
    // std::vector<std::string> supported_or_unsupported_op_types = {
    //    "batch_norm", "conv2d", "pool2d"};
    // bool is_supported_op_types = true;
    bool found = std::find(supported_or_unsupported_op_types.begin(),
                           supported_or_unsupported_op_types.end(),
                           op_type) != supported_or_unsupported_op_types.end();
    return (is_supported_op_types && found) ||
           (!is_supported_op_types && !found);
  };

  std::vector<std::vector<Node*>> subgraphs =
      SubgraphDetector(graph.get(), teller)();
  SubgraphVisualizer(graph.get(), subgraphs)();

  for (int subgraph_idx = 0; subgraph_idx < subgraphs.size(); subgraph_idx++) {
    InsertNewNode(graph.get(), subgraph_idx, subgraphs[subgraph_idx]);
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(subgraph_pass, paddle::lite::mir::SubgraphPass)
    .BindTargets({TARGET(kNPU), TARGET(kXPU)});
