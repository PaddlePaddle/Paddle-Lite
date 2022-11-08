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

#include "lite/core/optimizer/mir/generate_program_pass.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/optimizer/mir/graph_visualize_pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void GenerateProgramPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  VLOG(4) << "final program \n" << Visualize(graph.get());
  std::vector<Node*> nodes_in_order;
  if (nodes_in_order.empty()) {
    nodes_in_order = graph->StmtTopologicalOrder();
  }

  insts_.emplace_back();
  for (auto& item : nodes_in_order) {
    if (item->IsStmt()) {
      auto& stmt = item->AsStmt();
      VLOG(4) << stmt;
      insts_.back().emplace_back(stmt.op(), std::move(stmt.kernels().front()));
    }
  }

  // Update precision info after opt optimizations are operated.
  std::vector<std::string> skip_ops = {
      "while", "conditional_block", "feed", "fetch"};

  for (auto& node : nodes_in_order) {
    auto op_type = node->AsStmt().op_type();
    auto iter = std::find(skip_ops.begin(), skip_ops.end(), op_type);
    if (!node->IsStmt() || iter != skip_ops.end()) continue;
    // complement inputs precisions
    auto inlinks = node->inlinks;
    for (auto* in : inlinks) {
      // Create the new var manually.
      auto in_arg_name = in->AsArg().name;
      if (!(in->AsArg().is_weight) && in->AsArg().type->IsTensor()) {
        auto* tmp_tensor = node->AsStmt()
                               .op()
                               ->scope()
                               ->Var(in_arg_name)
                               ->GetMutable<lite::Tensor>();
        if ((tmp_tensor->precision() != in->AsArg().type->precision())) {
          tmp_tensor->set_precision(in->AsArg().type->precision());
        }
      }
    }
  }
}
}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(generate_program_pass, paddle::lite::mir::GenerateProgramPass)
    .BindTargets({TARGET(kAny)});
