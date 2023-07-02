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

#include "lite/core/optimizer/mir/__xpu__quantization_parameters_propagation_pass.h"
#include <cmath>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "lite/core/optimizer/mir/graph_visualize_pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

static bool HasOutputThreshold(const OpInfo* op_info,
                               const std::string& name,
                               bool is_threshold_name) {
  bool res = false;
  if (is_threshold_name) {
    res = op_info->HasAttr(name);
  } else {
    std::string argname;
    int index;
    if (op_info->HasAttr("out_threshold")) {
      res = true;
    } else if (op_info->GetOutputArgname(name, &argname) &&
               op_info->GetOutputIndex(name, &index)) {
      res = op_info->HasAttr(argname + to_string(index) + "_threshold");
    }
  }
  return res;
}

static float GetOutputThreshold(const OpInfo* op_info,
                                const std::string& name,
                                bool is_threshold_name) {
  std::string threshold_name;
  if (is_threshold_name) {
    threshold_name = name;
  } else if (op_info->HasAttr("out_threshold")) {
    threshold_name = "out_threshold";
  } else {
    std::string arg_name;
    int arg_index;
    CHECK(op_info->GetOutputArgname(name, &arg_name));
    CHECK(op_info->GetOutputIndex(name, &arg_index));
    threshold_name = arg_name + to_string(arg_index) + "_threshold";
  }
  float threshold = 0.0;
  if (op_info->GetAttrType(threshold_name) == OpDescAPI::AttrType::FLOAT) {
    threshold = op_info->GetAttr<float>(threshold_name);
  } else if (op_info->GetAttrType(threshold_name) ==
             OpDescAPI::AttrType::FLOATS) {
    threshold = op_info->GetAttr<std::vector<float>>(threshold_name)[0];
  }
  return threshold;
}

// Complete the output scale from the input scale of its consumer ops.
static bool SetOutScaleFromNextInScale(
    const std::unique_ptr<SSAGraph>& graph,
    int auto_complete_quant_scale_level = 0) {
  bool found = false;
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().mutable_op_info();
    for (auto in_var_node : op_node->inlinks) {
      CHECK(in_var_node->IsArg());
      auto in_var_name = in_var_node->arg()->name;
      if (!op_info->HasInputScale(in_var_name)) continue;
      auto in_var_scale = op_info->GetInputScale(in_var_name);
      for (auto in_op_node : in_var_node->inlinks) {
        if (!in_op_node->IsStmt()) continue;
        auto in_op_info = in_op_node->AsStmt().mutable_op_info();
        bool in_op_is_quanted =
            HasOutputThreshold(in_op_info, in_var_name, false);
        auto in_op_in_vars = in_op_node->inlinks;
        for (auto in_op_in_var : in_op_in_vars) {
          in_op_is_quanted =
              in_op_is_quanted ||
              in_op_info->HasInputScale(in_op_in_var->arg()->name);
        }
        in_op_is_quanted =
            in_op_is_quanted || auto_complete_quant_scale_level >= 1;
        if (in_op_is_quanted) {
          // Use this input scale to update the output scale of the quantized op
          in_op_info->SetOutputScale(in_var_name, in_var_scale);
          found = true;
        }
      }
    }
  }
  return found;
}

// Complete the output scale from the user-defined configurations.
static bool SetOutScaleFromConfigs(
    const std::unique_ptr<SSAGraph>& graph,
    const std::string& auto_complete_quant_scale_configs) {
  if (auto_complete_quant_scale_configs.empty()) return false;
  std::unordered_map<std::string, float> var_nodes;
  const auto& lines = Split(auto_complete_quant_scale_configs, "\n");
  for (const auto& line : lines) {
    const auto& items = Split(line, "-");
    if (items.empty()) continue;
    for (const auto& item : items) {
      if (item.empty()) continue;
      const auto& vars = Split(item, ",");
      for (const auto& var : vars) {
        auto info = Split(var, ":");
        CHECK_EQ(info.size(), 2);
        auto& out_var_name = info[0];
        auto out_threshold = parse_string<float>(info[1]);
        CHECK_GT(out_threshold, 0);
        CHECK(!var_nodes.count(out_var_name));
        var_nodes[out_var_name] = out_threshold;
      }
    }
  }
  if (var_nodes.empty()) return false;
  bool found = false;
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().mutable_op_info();
    for (auto out_var_node : op_node->outlinks) {
      CHECK(out_var_node->IsArg());
      auto out_var_name = out_var_node->arg()->name;
      if (op_info->HasOutputScale(out_var_name)) continue;
      if (!var_nodes.count(out_var_name)) continue;
      int bit_length = 8;  // op_info->GetAttr<int>("bit_length");
      int range = (1 << (bit_length - 1)) - 1;
      auto out_var_scale =
          std::vector<float>{var_nodes.at(out_var_name) / range};
      op_info->SetOutputScale(out_var_name, out_var_scale);
      found = true;
    }
  }
  return found;
}

// Complete the output scale from its out_threshold attribute.
static bool SetOutScaleFromCurOutThreshold(
    const std::unique_ptr<SSAGraph>& graph) {
  bool found = false;
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().mutable_op_info();
    for (auto out_var_node : op_node->outlinks) {
      CHECK(out_var_node->IsArg());
      auto out_var_name = out_var_node->arg()->name;
      if (op_info->HasOutputScale(out_var_name))
        continue;  // skip if the output scale had been already set
      if (!HasOutputThreshold(op_info, out_var_name, false)) continue;
      // If it's a quantized op, calculate its output scale according to the
      // output threshold(abs_max value)
      int bit_length = 8;  // op_info->GetAttr<int>("bit_length");
      int range = (1 << (bit_length - 1)) - 1;
      auto out_var_scale = std::vector<float>{
          GetOutputThreshold(op_info, out_var_name, false) / range};
      op_info->SetOutputScale(out_var_name, out_var_scale);
      found = true;
    }
  }
  return found;
}

// Complete the input scale from the output scale of its producer op.
static bool SetInScaleFromPrevOutScale(const std::unique_ptr<SSAGraph>& graph) {
  bool found = false;
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().mutable_op_info();
    for (auto in_var_node : op_node->inlinks) {
      CHECK(in_var_node->IsArg());
      auto in_var_name = in_var_node->arg()->name;
      if (op_info->HasInputScale(in_var_name)) continue;
      std::vector<float> in_var_scale;
      for (auto in_op_node : in_var_node->inlinks) {
        if (!in_op_node->IsStmt()) continue;
        auto in_op_info = in_op_node->AsStmt().mutable_op_info();
        if (!in_op_info->HasOutputScale(in_var_name)) continue;
        auto candidate_var_scale = in_op_info->GetOutputScale(in_var_name);
        if (!in_var_scale.empty()) {
          auto scale_size = in_var_scale.size();
          CHECK_EQ(scale_size, candidate_var_scale.size());
          for (size_t i = 0; i < scale_size; i++) {
            CHECK(fabs(in_var_scale[i] - candidate_var_scale[i]) <= 1e-7f);
          }
        } else {
          in_var_scale = candidate_var_scale;
        }
      }
      if (!in_var_scale.empty()) {
        op_info->SetInputScale(in_var_name, in_var_scale);
        found = true;
      }
    }
  }
  return found;
}

// Complete the output scale according to the input scale, because the input
// scale and output scale of the ops should be the same.
static bool SetOutScaleFromCurInScale(
    const std::unique_ptr<SSAGraph>& graph,
    const std::unordered_map<std::string,
                             std::unordered_map<std::string, std::string>>&
        op_types) {
  bool found = false;
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().mutable_op_info();
    auto op_type = op_info->Type();
    if (!op_types.count(op_type)) continue;
    for (auto out_var_node : op_node->outlinks) {
      CHECK(out_var_node->IsArg());
      auto out_var_name = out_var_node->arg()->name;
      if (op_info->HasOutputScale(out_var_name)) continue;
      std::string out_arg_name;
      if (!op_info->GetOutputArgname(out_var_name, &out_arg_name)) continue;
      for (auto in_var_node : op_node->inlinks) {
        CHECK(in_var_node->IsArg());
        auto in_var_name = in_var_node->arg()->name;
        if (!op_info->HasInputScale(in_var_name)) continue;
        std::string in_arg_name;
        if (!op_info->GetInputArgname(in_var_name, &in_arg_name)) continue;
        if (!op_types.at(op_type).count(in_arg_name)) continue;
        if (op_types.at(op_type).at(in_arg_name) != out_arg_name) continue;
        op_info->SetOutputScale(out_var_name,
                                op_info->GetInputScale(in_var_name));
        found = true;
        break;
      }
    }
  }
  return found;
}

// Complete the input scale according to the output scale, because the input
// scale and output scale of the ops should be the same.
static bool SetInScaleFromCurOutScale(
    const std::unique_ptr<SSAGraph>& graph,
    const std::unordered_map<std::string,
                             std::unordered_map<std::string, std::string>>&
        op_types) {
  bool found = false;
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().mutable_op_info();
    auto op_type = op_info->Type();
    if (!op_types.count(op_type)) continue;
    for (auto in_var_node : op_node->inlinks) {
      CHECK(in_var_node->IsArg());
      auto in_var_name = in_var_node->arg()->name;
      if (op_info->HasInputScale(in_var_name)) continue;
      std::string in_arg_name;
      if (!op_info->GetInputArgname(in_var_name, &in_arg_name)) continue;
      if (!op_types.at(op_type).count(in_arg_name)) continue;
      for (auto out_var_node : op_node->outlinks) {
        CHECK(out_var_node->IsArg());
        auto out_var_name = out_var_node->arg()->name;
        if (!op_info->HasOutputScale(out_var_name)) continue;
        std::string out_arg_name;
        if (!op_info->GetOutputArgname(out_var_name, &out_arg_name)) continue;
        if (op_types.at(op_type).at(in_arg_name) != out_arg_name) continue;
        op_info->SetInputScale(in_var_name,
                               op_info->GetOutputScale(out_var_name));
        found = true;
        break;
      }
    }
  }
  return found;
}

// Complete the output scale according to the formula of some special ops
// themselves.
static bool SetOutScaleFromSpecialOps(const std::unique_ptr<SSAGraph>& graph) {
  const std::unordered_set<std::string> op_types{"relu6", "softmax"};
  bool found = false;
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().mutable_op_info();
    auto op_type = op_info->Type();
    if (!op_types.count(op_type)) continue;
    for (auto out_var_node : op_node->outlinks) {
      CHECK(out_var_node->IsArg());
      auto out_var_name = out_var_node->arg()->name;
      if (op_info->HasOutputScale(out_var_name)) continue;
      std::string out_arg_name;
      if (!op_info->GetOutputArgname(out_var_name, &out_arg_name)) continue;
      float out_threshold;
      if (op_type == "relu6" && out_arg_name == "Out") {
        out_threshold = 6.0f;
      } else if (op_type == "softmax" && out_arg_name == "Out") {
        out_threshold = 1.0f;
      } else {
        continue;
      }
      int bit_length = 8;  // op_info->GetAttr<int>("bit_length");
      int range = (1 << (bit_length - 1)) - 1;
      auto out_var_scale = std::vector<float>{out_threshold / range};
      op_info->SetOutputScale(out_var_name, out_var_scale);
      found = true;
    }
  }
  return found;
}

// Print variables without outscale
static void PrintVariablesWithoutOutScale(
    const std::unique_ptr<SSAGraph>& graph) {
  std::ostringstream os;
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().mutable_op_info();
    auto op_type = op_info->Type();
    for (auto out_var_node : op_node->outlinks) {
      CHECK(out_var_node->IsArg());
      auto out_var_name = out_var_node->arg()->name;
      if (op_info->HasOutputScale(out_var_name)) continue;
      if (out_var_node->outlinks.size() > 0) os << out_var_name << "\n";
    }
  }
  VLOG(5) << "\nVariables without outscale:\n" << os.str();
}

void XPUQuantizationParametersPropagationPass::ResetScale(
    const std::unique_ptr<SSAGraph>& graph) {
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto op_info = op_node->AsStmt().mutable_op_info();
    auto op_type = op_info->Type();

    int bit_length = 8;  // op_info->GetAttr<int>("bit_length");
    int range = (1 << (bit_length - 1)) - 1;

    // Reset intput/output sclae value,thanks to lite quant pass.
    for (auto in_var_node : op_node->inlinks) {
      CHECK(in_var_node->IsArg());
      auto in_var_name = in_var_node->arg()->name;
      if (!op_info->HasInputScale(in_var_name)) continue;
      auto in_var_scale = op_info->GetInputScale(in_var_name);
      for (int i = 0; i < in_var_scale.size(); i++) {
        in_var_scale[i] = in_var_scale[i] * range;
      }
      op_info->SetInputScale(in_var_name, in_var_scale);
    }

    for (auto out_var_node : op_node->outlinks) {
      CHECK(out_var_node->IsArg());
      auto out_var_name = out_var_node->arg()->name;
      if (!op_info->HasOutputScale(out_var_name)) continue;
      auto out_var_scale = op_info->GetOutputScale(out_var_name);
      for (int i = 0; i < out_var_scale.size(); i++) {
        out_var_scale[i] = out_var_scale[i] * range;
      }
      op_info->SetOutputScale(out_var_name, out_var_scale);
    }
  }
}

void XPUQuantizationParametersPropagationPass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  VLOG(5) << "\n" << Visualize(graph.get());
  // Due to various reasons (such as bugs from PaddleSlim), some ops in
  // the
  // model lack quantization parameters. Optionally, the missing
  // quantization
  // parameters can be completed by the following rules.

  auto auto_complete_quant_scale_level =
      GetIntFromEnv(QUANT_AUTO_COMPLETE_SCALE_LEVEL);
  // (a) Complete the output scale from the input scale of its consumer
  // ops.
  SetOutScaleFromNextInScale(graph, auto_complete_quant_scale_level);

  // (b) Complete the output scale from the user -
  //     defined configurations.
  auto auto_complete_quant_scale_configs =
      GetConfigsFromEnv(QUANT_AUTO_COMPLETE_SCALE_CONFIG_FILE,
                        QUANT_AUTO_COMPLETE_SCALE_CONFIG_BUFFER);
  if (!auto_complete_quant_scale_configs.empty()) {
    SetOutScaleFromConfigs(graph, auto_complete_quant_scale_configs);
  }

  // (c) Complete the output scale from its out_threshold attribute.
  SetOutScaleFromCurOutThreshold(graph);

  // (d) Complete the input scale from the output scale of its producer
  // op.
  SetInScaleFromPrevOutScale(graph);

  // (e) Reset all scale(* 127) in xpu op.
  ResetScale(graph);

  // (f) Complete the output scale according to the input scale, or
  // complete
  // the input scale according to the output scale, because the input
  // scale and
  // output scale of some ops should be the same.
  const std::unordered_map<std::string,
                           std::unordered_map<std::string, std::string>>
      in_scale_same_as_out_scale_ops{
          {"transpose", {{"X", "Out"}}},
          {"transpose2", {{"X", "Out"}}},
          {"squeeze", {{"X", "Out"}}},
          {"squeeze2", {{"X", "Out"}}},
          {"unsqueeze", {{"X", "Out"}}},
          {"unsqueeze2", {{"X", "Out"}}},
          {"reshape", {{"X", "Out"}}},
          {"reshape2", {{"X", "Out"}}},
          {"flatten", {{"X", "Out"}}},
          {"flatten2", {{"X", "Out"}}},
          {"flatten_contiguous_range", {{"X", "Out"}}},
          {"expand", {{"X", "Out"}}},
          {"expand_v2", {{"X", "Out"}}},
          {"bilinear_interp", {{"X", "Out"}}},
          {"bilinear_interp_v2", {{"X", "Out"}}},
          {"nearest_interp", {{"X", "Out"}}},
          {"nearest_interp_v2", {{"X", "Out"}}},
          {"pool2d", {{"X", "Out"}}},
          {"leaky_relu", {{"X", "Out"}}},
          {"relu", {{"X", "Out"}}}};
  if (auto_complete_quant_scale_level >= 2) {
    bool found = true;
    do {
      found = SetOutScaleFromCurInScale(graph, in_scale_same_as_out_scale_ops);
      SetInScaleFromPrevOutScale(graph);
    } while (found);
    do {
      found = SetInScaleFromCurOutScale(graph, in_scale_same_as_out_scale_ops);
      SetOutScaleFromNextInScale(graph, auto_complete_quant_scale_level);
    } while (found);
  }
  // (g) Complete the output scale according to the formula of some
  // special ops
  // themselves.
  if (auto_complete_quant_scale_level >= 3) {
    SetOutScaleFromSpecialOps(graph);
    SetInScaleFromPrevOutScale(graph);
    bool found = true;
    do {
      found = SetOutScaleFromCurInScale(graph, in_scale_same_as_out_scale_ops);
      SetInScaleFromPrevOutScale(graph);
    } while (found);
    do {
      found = SetInScaleFromCurOutScale(graph, in_scale_same_as_out_scale_ops);
      SetOutScaleFromNextInScale(graph, auto_complete_quant_scale_level);
    } while (found);
  }

  // Print variables without outscale to help users set out threshold
  // manually
  PrintVariablesWithoutOutScale(graph);
  VLOG(5) << "\n" << Visualize(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__quantization_parameters_propagation_pass,
                  paddle::lite::mir::XPUQuantizationParametersPropagationPass)
    .BindTargets({TARGET(kXPU)});
