// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/optimizer/mir/elimination/utility.h"
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

void CollectControlFlowOpInputsOutputs(
    const std::unique_ptr<mir::SSAGraph>& graph,
    std::unordered_set<std::string>* in_vars,
    std::unordered_set<std::string>* out_vars) {
  const std::unordered_set<std::string> control_flow_op_types = {
      "while", "conditional_block"};
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().op_info();
    auto op_type = op_info->Type();
    if (control_flow_op_types.count(op_type)) {
      for (auto& var_node : op_node->inlinks) {
        auto& var_name = var_node->AsArg().name;
        if (!in_vars->count(var_name)) {
          in_vars->insert(var_name);
        }
      }
      for (auto& var_node : op_node->outlinks) {
        auto& var_name = var_node->AsArg().name;
        if (!out_vars->count(var_name)) {
          out_vars->insert(var_name);
        }
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
