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

#include "lite/core/optimizer/mir/elimination/assign_value_calc_offline_pass.h"
#include <algorithm>
#include <cmath>
#include <list>
#include <memory>
#include <set>
#include <vector>
#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher.h"
#include "lite/core/optimizer/mir/ssa_graph_utils.h"
#include "lite/model_parser/cpp_desc.h"

namespace paddle {
namespace lite {
namespace mir {

void AssignValueCalcOfflinePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  RemoveAssignValuePattern(graph);
}

void AssignValueCalcOfflinePass::RemoveAssignValuePattern(
    const std::unique_ptr<SSAGraph>& graph) {
  for (auto& node : graph->StmtTopologicalOrder()) {
    if (node->AsStmt().op_type() != "assign_value") continue;
    auto outlinks = node->outlinks;
    bool has_extra_producers = false;
    for (auto& out_link : outlinks) {
      if (HasExtraProducers(
              graph.get(), out_link->arg()->name, {"assign_value"})) {
        has_extra_producers = true;
        break;
      }
    }
    if (has_extra_producers) {
      LOG(WARNING)
          << "The output var of op is not supported with multiple producers";
      continue;
    }

    std::set<const Node*> nodes2rm_;
    auto& assign_value_instruct = node->AsStmt();
    auto* scope = assign_value_instruct.op()->scope();
    auto op_desc = assign_value_instruct.mutable_op_info();

    // Get assign_value's attr
    CHECK(op_desc->HasAttr("shape"));
    auto shape = op_desc->GetAttr<std::vector<int>>("shape");
    CHECK(op_desc->HasAttr("dtype"));
    auto dtype = op_desc->GetAttr<int>("dtype");

    // Get assign_value's output tensor
    auto out_var = scope->FindVar(op_desc->Output("Out").front());
    auto out_t = out_var->GetMutable<lite::Tensor>();
    std::vector<int64_t> shape_int64_t;
    for (auto value : shape) {
      shape_int64_t.push_back(static_cast<int64_t>(value));
    }
    out_t->Resize(DDim(shape_int64_t));
    auto out_data = out_t->mutable_data<float>();

    if (dtype == static_cast<int>(lite::core::FluidType::INT32)) {
      auto int32_values = op_desc->GetAttr<std::vector<int>>("int32_values");
      memcpy(out_data, int32_values.data(), sizeof(int) * int32_values.size());
    } else if (dtype == static_cast<int>(lite::core::FluidType::FP32)) {
      auto fp32_values = op_desc->GetAttr<std::vector<float>>("fp32_values");
      memcpy(out_data, fp32_values.data(), sizeof(float) * fp32_values.size());
    } else if (dtype == static_cast<int>(lite::core::FluidType::INT64)) {
      auto int64_values =
          op_desc->GetAttr<std::vector<int64_t>>("int64_values");
      memcpy(
          out_data, int64_values.data(), sizeof(int64_t) * int64_values.size());
    } else if (dtype == static_cast<int>(lite::core::FluidType::BOOL)) {
      auto bool_values = op_desc->GetAttr<std::vector<int>>("bool_values");
      memcpy(out_data, bool_values.data(), sizeof(bool) * bool_values.size());
    } else {
      LOG(FATAL) << "Unsupported dtype for assign_value op: " << dtype;
    }

    // Offline calc assign_value, only retain output tensor as persistable
    // tensor
    out_t->set_persistable(true);
    auto assign_value_outlinks = node->outlinks;
    for (auto& assign_value_out_link : assign_value_outlinks) {
      assign_value_out_link->arg()->is_weight = true;
    }
    nodes2rm_.insert(node);
    GraphSafeRemoveNodes(graph.get(), nodes2rm_);
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(assign_value_calc_offline_pass,
                  paddle::lite::mir::AssignValueCalcOfflinePass)
    .BindTargets({TARGET(kNNAdapter)});
