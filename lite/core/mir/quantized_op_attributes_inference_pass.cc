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

#include "lite/core/mir/quantized_op_attributes_inference_pass.h"
#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void QuantizedOpAttributesInferencePass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  // Only for fully quantized model which is only supported by MTK and RK NPU.
  // Replace the output_scale with the input_scale of the adjacent quantized
  // ops, and fix the missing of the attribute 'enable_int8'.
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto& inst = op_node->AsStmt();
    auto op_info = inst.op_info();
    auto op_type = op_info->Type();
    if (!op_info->HasAttr("input_scale")) continue;
    bool found = false;
    float output_scale;
    for (auto out_var_node : op_node->outlinks) {
      CHECK(out_var_node->IsArg());
      for (auto out_op_node : out_var_node->outlinks) {
        CHECK(out_op_node->IsStmt());
        auto& out_inst = out_op_node->AsStmt();
        auto out_op_info = out_inst.op_info();
        if (!out_op_info->HasAttr("input_scale")) continue;
        auto input_scale = out_op_info->GetAttr<float>("input_scale");
        if (!found) {
          found = true;
          output_scale = input_scale;
        } else {
          CHECK_EQ(output_scale, input_scale);
        }
      }
    }
    if (found) {
      inst.mutable_op_info()->SetAttr("output_scale", output_scale);
    } else if (op_info->HasAttr("output_scale")) {
      int bit_length = op_info->GetAttr<int>("bit_length");
      int range = (1 << (bit_length - 1)) - 1;
      output_scale = op_info->GetAttr<float>("output_scale");
      inst.mutable_op_info()->SetAttr("output_scale", output_scale / range);
    }
    if (op_info->HasAttr("output_scale")) {
      inst.mutable_op_info()->SetAttr("enable_int8", true);
    }
  }
  VLOG(5) << "\n" << Visualize(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(quantized_op_attributes_inference_pass,
                  paddle::lite::mir::QuantizedOpAttributesInferencePass)
    .BindTargets({TARGET(kAPU), TARGET(kRKNPU)});
