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

#include "lite/core/mir/restrict_quantized_op_with_same_input_output_scale_pass.h"
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

// Collect the input/output scales and the touched op nodes recursively
void CollectInputOutputScales(
    Node* op_node,
    const std::unordered_set<std::string>& restricted_quantized_op_types,
    std::unordered_set<Node*>* touched_op_nodes,
    std::unordered_map<std::string, float>* in_out_scales) {
  if (!op_node->IsStmt()) return;
  if (touched_op_nodes->count(op_node)) return;
  touched_op_nodes->insert(op_node);
  auto op_info = op_node->AsStmt().op_info();
  auto op_type = op_info->Type();
  if (!restricted_quantized_op_types.count(op_type)) return;
  for (auto* in_var_node : op_node->inlinks) {
    CHECK(in_var_node->IsArg());
    auto in_var_name = in_var_node->arg()->name;
    CHECK(op_info->HasInputScale(in_var_name));
    auto in_scales = op_info->GetInputScale(in_var_name);
    CHECK_EQ(in_scales.size(), 1);
    auto in_scale = in_scales[0];
    if (in_out_scales->count(in_var_name)) {
      CHECK_EQ(in_out_scales->at(in_var_name), in_scale);
    } else {
      (*in_out_scales)[in_var_name] = in_scale;
    }
    for (auto* in_op_node : in_var_node->inlinks) {
      CollectInputOutputScales(in_op_node,
                               restricted_quantized_op_types,
                               touched_op_nodes,
                               in_out_scales);
    }
  }
  for (auto* out_var_node : op_node->outlinks) {
    CHECK(out_var_node->IsArg());
    auto out_var_name = out_var_node->arg()->name;
    CHECK(op_info->HasOutputScale(out_var_name));
    auto out_scales = op_info->GetOutputScale(out_var_name);
    CHECK_EQ(out_scales.size(), 1);
    auto out_scale = out_scales[0];
    if (in_out_scales->count(out_var_name)) {
      CHECK_EQ(in_out_scales->at(out_var_name), out_scale);
    } else {
      (*in_out_scales)[out_var_name] = out_scale;
    }
    for (auto* out_op_node : out_var_node->outlinks) {
      CollectInputOutputScales(out_op_node,
                               restricted_quantized_op_types,
                               touched_op_nodes,
                               in_out_scales);
    }
  }
}

// Update all related quantized op with the new scale
void UpdateInputOutputScales(
    const std::unordered_set<Node*>& touched_op_nodes,
    const std::unordered_map<std::string, float>& in_out_scales) {
  for (auto* op_node : touched_op_nodes) {
    CHECK(op_node->IsStmt());
    auto op_info = op_node->AsStmt().mutable_op_info();
    auto op_type = op_info->Type();
    for (auto* in_var_node : op_node->inlinks) {
      CHECK(in_var_node->IsArg());
      auto in_var_name = in_var_node->arg()->name;
      if (in_out_scales.count(in_var_name)) {
        op_info->SetInputScale(in_var_name, {in_out_scales.at(in_var_name)});
      }
    }
    for (auto* out_var_node : op_node->outlinks) {
      CHECK(out_var_node->IsArg());
      auto out_var_name = out_var_node->arg()->name;
      if (in_out_scales.count(out_var_name)) {
        op_info->SetOutputScale(out_var_name, {in_out_scales.at(out_var_name)});
      }
    }
  }
}

void RestrictQuantizedOpWithSameInputOutputScalePass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  VLOG(5) << "\n" << Visualize(graph.get());
  const std::unordered_set<std::string> restricted_quantized_op_types = {
      "concat"};
  int restrict_method =
      GetIntFromEnv(QUANT_INPUT_OUTPUT_SCALE_RESTRICT_METHOD, 0);
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    // Collect the input/output scales and the touched op nodes recursively
    std::unordered_set<Node*> touched_op_nodes;
    std::unordered_map<std::string, float> in_out_scales;
    CollectInputOutputScales(op_node,
                             restricted_quantized_op_types,
                             &touched_op_nodes,
                             &in_out_scales);
    if (in_out_scales.size() == 0) continue;
    // Figure out the minimum, maximium and mean scale
    float min_scale = std::numeric_limits<float>::max();
    float max_scale = std::numeric_limits<float>::min();
    float mean_scale = 0.0f;
    for (const auto& in_out_scale : in_out_scales) {
      mean_scale += in_out_scale.second;
      if (in_out_scale.second < min_scale) {
        min_scale = in_out_scale.second;
      }
      if (in_out_scale.second > max_scale) {
        max_scale = in_out_scale.second;
      }
    }
    mean_scale /= in_out_scales.size();
    // Not need to adjust if the scales are all the same
    if (std::fabs(max_scale - min_scale) <= 1e-5) continue;
    // Set the minimum, maximum and mean scale as the new scale according to the
    // environment variable 'QUANT_INPUT_OUTPUT_SCALE_RESTRICT_METHOD'
    float new_scale = 0.0f;
    switch (restrict_method) {
      case 1:
        new_scale = max_scale;
        break;
      case 2:
        new_scale = min_scale;
        break;
      default:
        new_scale = mean_scale;
        break;
    }
    for (auto& in_out_scale : in_out_scales) {
      in_out_scale.second = new_scale;
    }
    UpdateInputOutputScales(touched_op_nodes, in_out_scales);
  }
  VLOG(5) << "\n" << Visualize(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(
    restrict_quantized_op_with_same_input_output_scale_pass,
    paddle::lite::mir::RestrictQuantizedOpWithSameInputOutputScalePass)
    .BindTargets({TARGET(kRKNPU), TARGET(kNNAdapter)});
