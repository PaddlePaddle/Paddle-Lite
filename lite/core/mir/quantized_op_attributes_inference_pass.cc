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
  VLOG(5) << "\n" << Visualize(graph.get());
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto& inst = op_node->AsStmt();
    auto op_info = inst.op_info();
    auto op_type = op_info->Type();

    // Check if any of the inputs of the op have scale value
    bool has_input_scale = false;
    for (auto in_var_node : op_node->inlinks) {
      CHECK(in_var_node->IsArg());
      auto in_var_node_name = in_var_node->arg()->name;
      has_input_scale |= op_info->HasInputScale(in_var_node_name);
    }
    if (!has_input_scale) continue;

    // Infer the output scale according to its out_threshold or the input scale
    // of its adjacent ops
    bool is_quantized = true;
    for (auto out_var_node : op_node->outlinks) {
      CHECK(out_var_node->IsArg());
      std::vector<float> output_scale;
      bool has_output_scale = false;
      auto out_var_node_name = out_var_node->arg()->name;
      for (auto out_op_node : out_var_node->outlinks) {
        CHECK(out_op_node->IsStmt());
        auto& out_inst = out_op_node->AsStmt();
        auto out_op_info = out_inst.op_info();
        if (!out_op_info->HasInputScale(out_var_node_name)) continue;
        auto input_scale = out_op_info->GetInputScale(out_var_node_name);
        if (!has_output_scale) {
          output_scale = input_scale;
          has_output_scale = true;
        } else {
          CHECK_EQ(output_scale.size(), input_scale.size());
        }
      }
      if (has_output_scale) {
        inst.mutable_op_info()->SetOutputScale(out_var_node_name, output_scale);
      } else if (op_info->HasAttr("out_threshold")) {
        // Only consider one output, there are only one out_threshold
        int bit_length = op_info->GetAttr<int>("bit_length");
        int range = (1 << (bit_length - 1)) - 1;
        output_scale = std::vector<float>{
            op_info->GetAttr<float>("out_threshold") / range};
        inst.mutable_op_info()->SetOutputScale(out_var_node_name, output_scale);
      } else {
        is_quantized = false;
      }
    }

    // Fix the missing of the attribute 'enable_int8'.
    if (is_quantized) {
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
