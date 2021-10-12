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
#include "lite/model_parser/cpp_desc.h"

namespace paddle {
namespace lite {
namespace mir {

template <typename T>
void GetSize(T start, T end, T step, int64_t* size) {
  *size = std::is_integral<T>::value
              ? ((std::abs(end - start) + std::abs(step) - 1) / std::abs(step))
              : std::ceil(std::abs((end - start) / step));
}

void RangeCalcOfflinePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  RemoveRangePattern(graph);
}

void RangeCalcOfflinePass::RemoveRangePattern(
    const std::unique_ptr<SSAGraph>& graph) {
  for (auto& node : graph->StmtTopologicalOrder()) {
    if (node->AsStmt().picked_kernel().op_type() != "range") continue;

    std::set<const Node*> nodes2rm_;
    auto& range_instruct = node->AsStmt();
    auto* scope = range_instruct.op()->scope();
    auto op_desc = range_instruct.mutable_op_info();

    // Get range's input tensor
    LOG(INFO) << "[DEBUG]start input tensor ";
    auto start_var = scope->FindVar(op_desc->Input("Start").front());
    auto start_t = start_var->GetMutable<lite::Tensor>();
    auto start = start_t->mutable_data<float>()[0];
    LOG(INFO) << "[DEBUG]start data " << start;
    auto end_var = scope->FindVar(op_desc->Input("End").front());
    auto end_t = end_var->GetMutable<lite::Tensor>();
    auto end = end_t->mutable_data<float>()[0];
    LOG(INFO) << "[DEBUG]end data " << end;
    auto step_var = scope->FindVar(op_desc->Input("Step").front());
    auto step_t = step_var->GetMutable<lite::Tensor>();
    auto step = step_t->mutable_data<float>()[0];
    // Get range's output tensor
    auto out_var = scope->FindVar(op_desc->Output("Out").front());
    auto out_t = out_var->GetMutable<lite::Tensor>();

    // Calc range
    int64_t size = 0;
    GetSize(start, end, step, &size);

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
