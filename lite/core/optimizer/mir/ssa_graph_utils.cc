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

namespace paddle {
namespace lite {
namespace mir {

bool HasExtraProducers(mir::SSAGraph *graph,
                       const std::string &var_name,
                       const std::set<std::string> &exclude_op_list,
                       const std::set<std::string> &candidate_op) {
  for (auto &op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().op_info();
    auto op_type = op_info->Type();
    if (exclude_op_list.count(op_type)) continue;
    if (candidate_op.empty() || candidate_op.count(op_type)) {
      for (auto &var_node : op_node->outlinks) {
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

}  // namespace mir
}  // namespace lite
}  // namespace paddle
