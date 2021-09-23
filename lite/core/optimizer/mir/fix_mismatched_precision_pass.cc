// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/optimizer/mir/fix_mismatched_precision_pass.h"
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void FixMismatchedPrecisionPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  FixMismatchedPrecision(graph, "multiclass_nms2", "Index", PRECISION(kInt32));
  FixMismatchedPrecision(
      graph, "crf_decoding", "ViterbiPath", PRECISION(kInt64));
}

void FixMismatchedPrecisionPass::FixMismatchedPrecision(
    const std::unique_ptr<SSAGraph>& graph,
    const std::string target_op_type,
    const std::string target_arg_name,
    const lite_api::PrecisionType target_precision_type) {
  auto nodes = graph->StmtTopologicalOrder();
  for (auto& node : nodes) {
    if (!node->IsStmt()) continue;
    auto inst = node->stmt();
    if (inst->op_type() != target_op_type) continue;

    std::vector<Node*> io_nodes(node->inlinks.begin(), node->inlinks.end());
    io_nodes.insert(
        io_nodes.end(), node->outlinks.begin(), node->outlinks.end());
    for (auto& io_node : io_nodes) {
      auto arg = io_node->arg();
      std::string arg_name;
      CHECK(inst->op_info()->GetInputArgname(arg->name, &arg_name) ||
            inst->op_info()->GetOutputArgname(arg->name, &arg_name));
      if (arg_name == target_arg_name) {
        auto& arg_type = arg->type;
        if (arg_type->IsTensor()) {
          arg_type = LiteType::GetTensorTy(
              arg_type->target(), target_precision_type, arg_type->layout());
        } else if (arg_type->IsTensorList()) {
          arg_type = LiteType::GetTensorListTy(
              arg_type->target(), target_precision_type, arg_type->layout());
        } else {
          LOG(ERROR) << "unsupported arg type.";
        }
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(fix_mismatched_precision_pass,
                  paddle::lite::mir::FixMismatchedPrecisionPass)
    .BindTargets({TARGET(kXPU)});
