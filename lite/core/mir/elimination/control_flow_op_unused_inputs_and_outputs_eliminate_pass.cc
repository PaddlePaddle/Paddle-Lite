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

#include "lite/core/mir/elimination/control_flow_op_unused_inputs_and_outputs_eliminate_pass.h"
#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

// Remove all of the unused nodes from the contorl flow op and update the inputs
// and outputs of the op info The unused nodes are defined as the nodes which
// are only linked to the control flow op nodes but nerver linked to the other
// op nodes.
//
// For example:
// graph[0]: main block
//                      in_x
//             in_f      |   in_z(unused node)
//                  \    |    /
//                   \   |   /
//        in_w ------- while ------- in_y(unused_node)
//                    /  |
//                   /   |
// (unused node)out_y    |
//                     out_x
//
// graph[1]: sub block
//                     in_x
//                       |
//                       |
//                     conv2d----in_f
//                       |
//                       |
//                      fc ------in_w
//                       |
//                       |
//                     softmax
//                       |
//                       |
//                     out_x
//
// After the pass is applied:
//                      in_x
//             in_f      |
//                  \    |
//                   \   |
//        in_w ------- while
//                       |
//                       |
//                       |
//                     out_x

// Remove the var node from var2rm if it is recursively referred to any op in
// the subblock
void CollectUnusedInputOutputNodes(
    int block_idx,
    std::vector<std::unique_ptr<mir::SSAGraph>>* graphs,
    const std::unordered_set<std::string>& control_flow_op_types,
    std::unordered_map<std::string, Node*>* in_vars2rm,
    std::unordered_map<std::string, Node*>* out_vars2rm) {
  auto block_size = graphs->size();
  for (auto& op_node : (*graphs)[block_idx]->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().op_info();
    auto op_type = op_info->Type();
    if (control_flow_op_types.count(op_type)) {
      int sub_block_idx = op_info->GetAttr<int32_t>("sub_block");
      CHECK(block_idx >= 0 && block_idx < block_size);
      CollectUnusedInputOutputNodes(sub_block_idx,
                                    graphs,
                                    control_flow_op_types,
                                    in_vars2rm,
                                    out_vars2rm);
    } else {
      for (auto& var_node : op_node->inlinks) {
        auto& var_name = var_node->AsArg().name;
        if (in_vars2rm->count(var_name)) {
          in_vars2rm->erase(var_name);
        }
      }
      for (auto& var_node : op_node->outlinks) {
        auto& var_name = var_node->AsArg().name;
        // Tensor array may be only used as the output vars in the sublock
        if (in_vars2rm->count(var_name)) {
          in_vars2rm->erase(var_name);
        }
        if (out_vars2rm->count(var_name)) {
          out_vars2rm->erase(var_name);
        }
      }
    }
  }
}

// Remove the unused var nodes from the graph and update the op_info of the
// control flow op
void RemoveNodesFromGraphAndUpdateOpInfo(
    SSAGraph* graph,
    Node* op_node,
    const std::unordered_map<std::string, Node*>& in_vars2rm,
    const std::unordered_map<std::string, Node*>& out_vars2rm) {
  auto op_info = op_node->AsStmt().mutable_op_info();
  auto op_type = op_info->Type();
  // Unlink the in_vars2rm and out_vars2rm from the control flow op node, and
  // remove them if nerver used.
  for (auto& var_node : in_vars2rm) {
    VLOG(3) << "in var node '" << var_node.first << "' is unlinked to "
            << op_type;
    RemoveDirectedLink(var_node.second, op_node);
  }
  for (auto& var_node : out_vars2rm) {
    VLOG(3) << "out var node '" << var_node.first << "' is unlinked from "
            << op_type;
    RemoveDirectedLink(op_node, var_node.second);
    // Unlink from all of the out op nodes.
    std::unordered_set<Node*> out_op_nodes;
    for (auto* out_op_node : var_node.second->outlinks) {
      if (!out_op_nodes.count(out_op_node)) {
        out_op_nodes.insert(out_op_node);
      }
    }
    for (auto* out_op_node : out_op_nodes) {
      RemoveDirectedLink(var_node.second, out_op_node);
    }
  }
  // Remove the unused nodes from the graph if their inlinks and outlinks are
  // empty
  std::unordered_set<const Node*> removed_var_nodes;
  for (auto& var_node : in_vars2rm) {
    if (var_node.second->inlinks.empty() && var_node.second->outlinks.empty() &&
        !removed_var_nodes.count(var_node.second)) {
      removed_var_nodes.insert(var_node.second);
      graph->RemoveNode(var_node.second);
      VLOG(3) << "in var node " << var_node.first << " is removed";
    }
  }
  for (auto& var_node : out_vars2rm) {
    if (var_node.second->inlinks.empty() && var_node.second->outlinks.empty() &&
        !removed_var_nodes.count(var_node.second)) {
      removed_var_nodes.insert(var_node.second);
      graph->RemoveNode(var_node.second);
      VLOG(3) << "out var node " << var_node.first << " is removed";
    }
  }
  // Update the op info of the control flow op
  for (auto& input : *op_info->mutable_inputs()) {
    for (auto var = input.second.begin(); var != input.second.end();) {
      if (in_vars2rm.count(*var)) {
        var = input.second.erase(var);
      } else {
        ++var;
      }
    }
  }
  for (auto& output : *op_info->mutable_outputs()) {
    for (auto var = output.second.begin(); var != output.second.end();) {
      if (out_vars2rm.count(*var)) {
        var = output.second.erase(var);
      } else {
        ++var;
      }
    }
  }
}

void ControlFlowOpUnusedInputsAndOutputsEliminatePass::SetAllGraphs(
    std::vector<std::unique_ptr<mir::SSAGraph>>* graphs) {
  CHECK(graphs && !graphs->empty());
  graphs_ = graphs;
}

void ControlFlowOpUnusedInputsAndOutputsEliminatePass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  // Remove the unused input and output nodes from the control flow op nodes
  // Which are only linked to the control flow op nodes but nerver linked to the
  // other op nodes
  const std::unordered_set<std::string> control_flow_op_types = {
      "while", "conditional_block"};
  auto block_size = graphs_->size();
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().mutable_op_info();
    auto op_type = op_info->Type();
    if (!control_flow_op_types.count(op_type)) continue;
    int sub_block_idx = op_info->GetAttr<int32_t>("sub_block");
    CHECK(sub_block_idx >= 0 && sub_block_idx < block_size);
    // Initialize the unused nodes with all of the input and output nodes
    std::unordered_map<std::string, Node *> in_vars2rm, out_vars2rm;
    for (auto* var_node : op_node->inlinks) {
      auto& var_name = var_node->AsArg().name;
      if (!in_vars2rm.count(var_name)) {
        in_vars2rm.insert(std::pair<std::string, Node*>(var_name, var_node));
      }
    }
    for (auto* var_node : op_node->outlinks) {
      auto& var_name = var_node->AsArg().name;
      if (!out_vars2rm.count(var_name)) {
        out_vars2rm.insert(std::pair<std::string, Node*>(var_name, var_node));
      }
    }
    // Remove the nodes which used in subblock recursively, and the remaining
    // nodes are the unused one.
    CollectUnusedInputOutputNodes(sub_block_idx,
                                  graphs_,
                                  control_flow_op_types,
                                  &in_vars2rm,
                                  &out_vars2rm);
    if (in_vars2rm.size() > 0 || out_vars2rm.size() > 0) {
      // Remove the unused nodes from graph, and update the op info of the
      // control flow op
      RemoveNodesFromGraphAndUpdateOpInfo(
          graph.get(), op_node, in_vars2rm, out_vars2rm);
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(
    control_flow_op_unused_inputs_and_outputs_eliminate_pass,
    paddle::lite::mir::ControlFlowOpUnusedInputsAndOutputsEliminatePass)
    .BindTargets({TARGET(kNPU)});
