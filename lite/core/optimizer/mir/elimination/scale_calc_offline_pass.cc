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

#include "lite/core/optimizer/mir/elimination/scale_calc_offline_pass.h"
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

void ScaleCalcOfflinePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  RemoveScalePattern(graph);
}

void ScaleCalcOfflinePass::RemoveScalePattern(
    const std::unique_ptr<SSAGraph>& graph) {
  for (auto& node : graph->StmtTopologicalOrder()) {
    if (node->AsStmt().op_type() != "scale") continue;
    auto outlinks = node->outlinks;
    bool has_extra_producers = false;
    for (auto& out_link : outlinks) {
      if (HasExtraProducers(graph.get(), out_link->arg()->name, {"scale"})) {
        has_extra_producers = true;
        break;
      }
    }
    if (has_extra_producers) {
      LOG(WARNING)
          << "Unsupported for op output var containing multiple producers";
      continue;
    }

    std::set<const Node*> nodes2rm_;
    auto& scale_instruct = node->AsStmt();
    auto* scope = scale_instruct.op()->scope();
    auto op_desc = scale_instruct.mutable_op_info();
    // Get scale's input tensor
    auto x_var = scope->FindVar(op_desc->Input("X").front());
    auto x_t = x_var->GetMutable<lite::Tensor>();
    if (!x_t->persistable()) {
      LOG(WARNING) << "ScaleCalcOfflinePass does not support input that is not "
                      "persistable";
      continue;
    }
    auto x_data = x_t->mutable_data<float>();
    auto x_dims = x_t->dims();
    // Get scale's attr
    auto scale = op_desc->GetAttr<float>("scale");
    auto bias = op_desc->GetAttr<float>("bias");
    auto bias_after_scale = op_desc->GetAttr<bool>("bias_after_scale");
    if (!bias_after_scale) {
      bias *= scale;
    }
    // Get scale's output tensor
    auto out_var = scope->FindVar(op_desc->Output("Out").front());
    auto out_t = out_var->GetMutable<lite::Tensor>();
    out_t->Resize(x_dims);
    auto out_data = out_t->mutable_data<float>();
    for (int i = 0; i < x_dims.production(); i++) {
      out_data[i] = x_data[i] * scale + bias;
    }

    // Offline calc scale, only retain output tensor as persistable tensor
    out_t->set_persistable(true);
    auto scale_outlinks = node->outlinks;
    for (auto& scale_out_link : scale_outlinks) {
      scale_out_link->arg()->is_weight = true;
    }
    nodes2rm_.insert(node);
    GraphSafeRemoveNodes(graph.get(), nodes2rm_);
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(scale_calc_offline_pass,
                  paddle::lite::mir::ScaleCalcOfflinePass)
    .BindTargets({TARGET(kNNAdapter)});
