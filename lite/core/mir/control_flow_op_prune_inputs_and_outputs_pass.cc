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

#include "lite/core/mir/control_flow_op_prune_inputs_and_outputs_pass.h"
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

// Remove the var node from var2rm if it is recursively referred to any op in
// the subblock
void CollectInputsAndOutputs(
    int block_idx,
    std::vector<std::unique_ptr<mir::SSAGraph>>* graphs,
    std::unordered_map<std::string, Node*>* in_vars2rm,
    std::unordered_map<std::string, Node*>* out_vars2rm) {
  auto block_size = graphs->size();
  for (auto& op_node : (*graphs)[block_idx]->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().op_info();
    auto op_type = op_info->Type();
    if (op_type == "while" || op_type == "conditional_block") {
      int sub_block_idx = op_info->GetAttr<int32_t>("sub_block");
      CHECK(block_idx >= 0 && block_idx < block_size);
      CollectInputsAndOutputs(sub_block_idx, graphs, in_vars2rm, out_vars2rm);
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

// Remove the unused var nodes from the graph and update the inputs and outputs
// of op info
void PruneInputsAndOutputs(
    SSAGraph* graph,
    Node* op_node,
    const std::unordered_map<std::string, Node*>& in_vars2rm,
    const std::unordered_map<std::string, Node*>& out_vars2rm) {
  auto op_info = op_node->AsStmt().mutable_op_info();
  auto op_type = op_info->Type();
  // Unlink the in_vars2rm and out_vars2rm from the op node, and remove them if
  // nerver used.
  for (auto& var_node : in_vars2rm) {
    VLOG(3) << "in var node '" << var_node.first << "' is unlinked to "
            << op_type;
    RemoveDirectedLink(var_node.second, op_node);
  }
  for (auto& var_node : out_vars2rm) {
    VLOG(3) << "out var node '" << var_node.first << "' is unlinked from "
            << op_type;
    RemoveDirectedLink(op_node, var_node.second);
    // Unlink from all of out op nodes.
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
  std::unordered_set<const Node*> removed_var_nodes;
  for (auto& var_node : in_vars2rm) {
    if (var_node.second->inlinks.size() == 0 &&
        var_node.second->outlinks.size() == 0 &&
        !removed_var_nodes.count(var_node.second)) {
      removed_var_nodes.insert(var_node.second);
      graph->RemoveNode(var_node.second);
      VLOG(3) << "in var node " << var_node.first << " is removed";
    }
  }
  for (auto& var_node : out_vars2rm) {
    if (var_node.second->inlinks.size() == 0 &&
        var_node.second->outlinks.size() == 0 &&
        !removed_var_nodes.count(var_node.second)) {
      removed_var_nodes.insert(var_node.second);
      graph->RemoveNode(var_node.second);
      VLOG(3) << "out var node " << var_node.first << " is removed";
    }
  }
  // Update the inputs and outputs of op info
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

void ControlFlowOpPruneInputsAndOutputsPass::SetAllGraphs(
    std::vector<std::unique_ptr<mir::SSAGraph>>* graphs) {
  CHECK(graphs && !graphs->empty());
  graphs_ = graphs;
}

void ControlFlowOpPruneInputsAndOutputsPass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  // Prune the weight nodes which are only linked to while/conditional_block
  // op nodes but nerver linked to the other op nodes
  auto block_size = graphs_->size();
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().mutable_op_info();
    auto op_type = op_info->Type();
    if (op_type != "while" && op_type != "conditional_block") continue;
    int sub_block_idx = op_info->GetAttr<int32_t>("sub_block");
    CHECK(sub_block_idx >= 0 && sub_block_idx < block_size);
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
    CollectInputsAndOutputs(sub_block_idx, graphs_, &in_vars2rm, &out_vars2rm);
    if (in_vars2rm.size() > 0 || out_vars2rm.size() > 0) {
      PruneInputsAndOutputs(graph.get(), op_node, in_vars2rm, out_vars2rm);
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(control_flow_op_prune_inputs_and_outputs_pass,
                  paddle::lite::mir::ControlFlowOpPruneInputsAndOutputsPass)
    .BindTargets({TARGET(kNPU)});
