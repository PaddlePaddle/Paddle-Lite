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

#include "lite/core/optimizer/mir/ssa_graph_utils.h"
#include <vector>

namespace paddle {
namespace lite {
namespace mir {

bool HasExtraProducers(SSAGraph* graph,
                       const std::string& var_name,
                       const std::set<std::string>& exclude_op_list,
                       const std::set<std::string>& candidate_op) {
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().op_info();
    auto op_type = op_info->Type();
    if (exclude_op_list.count(op_type)) continue;
    if (candidate_op.empty() || candidate_op.count(op_type)) {
      for (auto& var_node : op_node->outlinks) {
        if (var_name == var_node->AsArg().name ||
            var_node->AsArg().name.find(std::string(var_name + "__Mangled_")) !=
                std::string::npos) {
          return true;
        }
      }
    }
  }
  return false;
}

std::set<Node*> GetNodesFromConfigs(SSAGraph* graph,
                                    const std::string& configs) {
  std::set<Node*> nodes;
  std::vector<std::string> lines = Split(configs, "\n");
  for (const auto& line : lines) {
    if (line.empty()) continue;
    std::vector<std::string> ops = Split(line, "-");
    for (const auto& op : ops) {
      if (op.empty()) continue;
      std::vector<std::string> info = Split(op, ":");
      std::string op_type = info.at(0);
      std::vector<std::string> in_vars_name;
      if (info.size() > 1) {
        in_vars_name = Split(info.at(1), ",");
      }
      std::vector<std::string> out_vars_name;
      if (info.size() > 2) {
        out_vars_name = Split(info.at(2), ",");
      }
      for (auto& node : graph->mutable_nodes()) {
        if (node.IsArg()) continue;
        auto stmt = node.stmt();
        if (op_type != stmt->op_type()) continue;
        auto in_var_nodes = node.inlinks;
        auto out_var_nodes = node.outlinks;
        if (in_vars_name.size() > in_var_nodes.size() ||
            out_vars_name.size() > out_var_nodes.size()) {
          continue;
        }
        bool matched = true;
        for (auto in_var_name : in_vars_name) {
          bool found = false;
          for (auto* in_var_node : in_var_nodes) {
            if (in_var_node->arg()->name == in_var_name) {
              found = true;
              break;
            }
          }
          if (!found) {
            matched = false;
            break;
          }
        }
        for (auto out_var_name : out_vars_name) {
          bool found = false;
          for (auto* out_var_node : out_var_nodes) {
            if (out_var_node->arg()->name == out_var_name) {
              found = true;
              break;
            }
          }
          if (!found) {
            matched = false;
            break;
          }
        }
        if (matched) {
          nodes.insert(&node);
        }
      }
    }
  }
  return nodes;
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
