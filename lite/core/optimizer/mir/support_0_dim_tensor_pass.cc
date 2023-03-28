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

#include "lite/core/optimizer/mir/support_0_dim_tensor_pass.h"
#include <list>
#include <memory>
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void Support0DimTensor::Apply(const std::unique_ptr<SSAGraph>& graph) {
  for (auto& x : graph->StmtTopologicalOrder()) {
    if (!x->IsStmt()) continue;
    auto& inst = x->AsStmt();
    VLOG(4) << "checking op " << inst.op_info()->Repr();

    std::string op_type = inst.op_type();
    auto op = inst.op();
    auto op_info = *x->stmt()->op_info();
    auto* scope = op->scope();

    if (inst.op_type() == "feed" || inst.op_type() == "fetch") {
      continue;
    }

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
        }
        op_info.SetAttr<std::vector<int64_t>>("shape", shape);
      }
      op->Attach(op_info, scope);
    }

    for (auto* in_node : x->inlinks) {
      auto& var = in_node->AsArg();
      const auto& var_name = var.name;
      std::string arg_name;
      CHECK(op_info.GetInputArgname(var_name, &arg_name))
          << "Can not find the input argument,current var name : " << var_name;
      auto tensor = scope->FindMutableTensor(var_name);
      auto dims = tensor->dims();

      if (tensor->persistable() && dims.empty()) {
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
