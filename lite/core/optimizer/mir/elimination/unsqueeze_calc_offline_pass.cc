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

#include "lite/core/optimizer/mir/elimination/unsqueeze_calc_offline_pass.h"
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

void UnsqueezeCalcOfflinePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  RemoveUnsqueezePattern(graph);
}

void UnsqueezeCalcOfflinePass::RemoveUnsqueezePattern(
    const std::unique_ptr<SSAGraph>& graph) {
  for (auto& node : graph->StmtTopologicalOrder()) {
    if (node->AsStmt().op_type() != "unsqueeze" &&
        node->AsStmt().op_type() != "unsqueeze2")
      continue;
    auto outlinks = node->outlinks;
    bool has_extra_producers = false;
    for (auto& out_link : outlinks) {
      if (HasExtraProducers(graph.get(),
                            out_link->arg()->name,
                            {"unsqueeze", "unsqueeze2"})) {
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
    auto& unsqueeze_instruct = node->AsStmt();
    auto* scope = unsqueeze_instruct.op()->scope();
    auto op_desc = unsqueeze_instruct.mutable_op_info();
    // Get unsqueeze's input tensor
    auto input_var = scope->FindVar(op_desc->Input("X").front());
    auto input_t = input_var->GetMutable<lite::Tensor>();
    if (!input_t->persistable()) {
      LOG(WARNING)
          << "UnsqueezeCalcOfflinePass does not support input that is not "
             "persistable";
      continue;
    }
    auto input_shape = input_t->dims().Vectorize();
    // Get unsqueeze's attr
    CHECK(op_desc->HasAttr("axes"));
    auto axes = op_desc->GetAttr<std::vector<int>>("axes");
    // Get unsqueeze's output tensor
    auto out_var = scope->FindVar(op_desc->Output("Out").front());
    auto out_t = out_var->GetMutable<lite::Tensor>();
    std::vector<int64_t> output_shape(input_shape);
    output_shape.insert(output_shape.end(), axes.size(), 1);
    out_t->CopyDataFrom(*input_t);
    uint32_t cur_size = input_shape.size();
    for (size_t i = 0; i < axes.size(); i++) {
      int32_t axis = axes[i] < 0 ? axes[i] + cur_size + 1 : axes[i];
      CHECK_GE(axis, 0);
      CHECK_LE(axis, cur_size);
      for (uint32_t j = cur_size; j > axis; j--) {
        output_shape[j] = output_shape[j - 1];
      }
      output_shape[axis] = 1;
      cur_size++;
    }
    out_t->Resize(DDim(output_shape));
    // Offline calc unsqueeze, only retain output tensor as persistable
    // tensor
    out_t->set_persistable(true);
    auto unsqueeze_outlinks = node->outlinks;
    for (auto& unsqueeze_out_link : unsqueeze_outlinks) {
      unsqueeze_out_link->arg()->is_weight = true;
    }
    nodes2rm_.insert(node);
    GraphSafeRemoveNodes(graph.get(), nodes2rm_);
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(unsqueeze_calc_offline_pass,
                  paddle::lite::mir::UnsqueezeCalcOfflinePass)
    .BindTargets({TARGET(kNNAdapter)});
