// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/optimizer/mir/support_0_dim_tensor_pass.h"
#include <list>
#include <memory>
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void Support0DimTensor::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // fix attr
  const std::vector<std::string> op_cases_fix_attr{"fill_constant",
                                                   "uniform_random",
                                                   "expand_v2",
                                                   "assign_value",
                                                   "gaussian_random",
                                                   "set_value"};
  for (auto& x : graph->StmtTopologicalOrder()) {
    if (!x->IsStmt()) continue;

    auto& inst = x->AsStmt();
    VLOG(4) << "checking op " << inst.op_info()->Repr();
    auto op = inst.op();
    auto op_info = *x->stmt()->op_info();
    std::string op_type = inst.op_type();
    auto* scope = op->scope();
    if (std::find(op_cases_fix_attr.begin(),
                  op_cases_fix_attr.end(),
                  op_type) != op_cases_fix_attr.end()) {
      if (op_info.HasAttr("shape")) {
        auto type = op_info.GetAttrType("shape");
        if (type == OpAttrType::INTS) {
          auto shape = op_info.GetAttr<std::vector<int32_t>>("shape");
          if (shape.empty()) {
            shape.push_back(1);
          }
          op_info.SetAttr<std::vector<int32_t>>("shape", shape);
        } else {
          auto shape = op_info.GetAttr<std::vector<int64_t>>("shape");
          if (shape.empty()) {
            shape.push_back(1);
            VLOG(4) << "op " << op_type
                    << ", shape dims empty, fix dim -> {1} ";
          }
          op_info.SetAttr<std::vector<int64_t>>("shape", shape);
        }
        op->Attach(op_info, scope);
      }
    }
  }

  // fix var node shape
  for (auto& node : graph->mutable_nodes()) {
    if (!node.IsArg()) continue;

    auto& var = node.AsArg();
    const auto& var_name = var.name;
    if (var_name == "feed" || var_name == "fetch") continue;

    for (auto* in : node.inlinks) {
      if (!in->IsStmt()) continue;

      auto* scope = in->AsStmt().op()->scope();
      auto* var_ptr = scope->FindVar(var_name);
      if (var_ptr == nullptr) {
        VLOG(4) << "Can't find ouput var_name:  " << var_name
                << "in current scope.";
        continue;
      }
      if (!var_ptr->IsType<lite::Tensor>()) {
        continue;
      }

      auto tensor = scope->FindMutableTensor(var_name);
      auto dims = tensor->dims();
      if (dims.empty()) {
        VLOG(4) << "ARG " << var_name << " dims empty, fix dim -> {1} ";
        tensor->Resize({1});
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(support_0_dim_tensor_pass,
                  paddle::lite::mir::Support0DimTensor)
    .BindTargets({TARGET(kAny)});
