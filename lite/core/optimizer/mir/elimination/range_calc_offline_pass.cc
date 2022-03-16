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

#include "lite/core/optimizer/mir/elimination/range_calc_offline_pass.h"
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

template <typename T>
int64_t GetSpanCount(T start, T end, T step) {
  return std::is_integral<T>::value
             ? ((std::abs(end - start) + std::abs(step) - 1) / std::abs(step))
             : std::ceil(std::abs((end - start) / step));
}

void RangeCalcOfflinePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  RemoveRangePattern(graph);
}

void RangeCalcOfflinePass::RemoveRangePattern(
    const std::unique_ptr<SSAGraph>& graph) {
  for (auto& node : graph->StmtTopologicalOrder()) {
    if (node->AsStmt().op_type() != "range") continue;
    auto outlinks = node->outlinks;
    bool has_extra_producers = false;
    for (auto& out_link : outlinks) {
      if (HasExtraProducers(graph.get(), out_link->arg()->name, {"range"})) {
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
    auto& range_instruct = node->AsStmt();
    auto* scope = range_instruct.op()->scope();
    auto op_desc = range_instruct.mutable_op_info();

    // Get range's input tensor
    auto start_var = scope->FindVar(op_desc->Input("Start").front());
    auto end_var = scope->FindVar(op_desc->Input("End").front());
    auto step_var = scope->FindVar(op_desc->Input("Step").front());
    auto start_t = start_var->GetMutable<lite::Tensor>();
    auto end_t = end_var->GetMutable<lite::Tensor>();
    auto step_t = step_var->GetMutable<lite::Tensor>();
    if (!start_t->persistable() || !end_t->persistable() ||
        !step_t->persistable()) {
      LOG(WARNING) << "RangeCalcOfflinePass does not support input that is not "
                      "persistable";
      continue;
    }
    auto start = start_t->mutable_data<float>()[0];
    auto end = end_t->mutable_data<float>()[0];
    auto step = step_t->mutable_data<float>()[0];
    // Get range's output tensor
    auto out_var = scope->FindVar(op_desc->Output("Out").front());
    auto out_t = out_var->GetMutable<lite::Tensor>();

    // Calc range
    int64_t size = GetSpanCount(start, end, step);

    out_t->Resize(DDim({size}));
    auto out_data = out_t->mutable_data<float>();

    float value = start;
    for (int64_t i = 0; i < size; ++i) {
      out_data[i] = value;
      value += step;
    }
    // Offline calc range, only retain output tensor as persistable tensor
    out_t->set_persistable(true);
    auto range_outlinks = node->outlinks;
    for (auto& range_out_link : range_outlinks) {
      range_out_link->arg()->is_weight = true;
    }
    nodes2rm_.insert(node);
    GraphSafeRemoveNodes(graph.get(), nodes2rm_);
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(range_calc_offline_pass,
                  paddle::lite::mir::RangeCalcOfflinePass)
    .BindTargets({TARGET(kNNAdapter)});
