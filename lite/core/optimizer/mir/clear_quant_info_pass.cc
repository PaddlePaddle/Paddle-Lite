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

#include "lite/core/optimizer/mir/clear_quant_info_pass.h"
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void ClearQuantInfoPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  Scope* scope = nullptr;
  for (auto& node : graph->nodes()) {
    if (node.IsStmt()) {
      scope = node.stmt()->op()->scope();
      break;
    }
  }
  CHECK(scope);

  std::string mixed_precision_quantization_config =
      GetMixedPrecisionQuantizationConfig(scope);
  if (mixed_precision_quantization_config.empty()) {
    VLOG(3) << "not receive mixed precision quantization config.";
    return;
  }
  std::set<Node*> target_nodes =
      GetTargetNodesFromMixedPrecisionQuantizationConfig(
          graph, mixed_precision_quantization_config);
  for (auto node : target_nodes) {
    CHECK(node->IsStmt());
    ClearQuantInfo(node);
  }
}

std::string ClearQuantInfoPass::GetMixedPrecisionQuantizationConfig(
    Scope* scope) {
  std::string mixed_precision_quantization_config;
#if defined(LITE_ON_MODEL_OPTIMIZE_TOOL) || defined(LITE_WITH_PYTHON) || \
    defined(LITE_WITH_NNADAPTER)
  mixed_precision_quantization_config = Context<TargetType::kNNAdapter>::
      NNAdapterMixedPrecisionQuantizationConfigBuffer(scope);
  if (!mixed_precision_quantization_config.empty()) {
    VLOG(3) << "Load mixed precision quantization config from buffer.";
  } else {
    auto path = Context<TargetType::kNNAdapter>::
        NNAdapterMixedPrecisionQuantizationConfigPath(scope);
    if (!path.empty()) {
      std::vector<char> buffer;
      if (ReadFile(path, &buffer, false)) {
        if (!buffer.empty()) {
          mixed_precision_quantization_config.insert(
              mixed_precision_quantization_config.begin(),
              buffer.begin(),
              buffer.end());
          VLOG(3) << "Load mixed precision quantization config from file: "
                  << mixed_precision_quantization_config;
        }
      } else {
        LOG(WARNING) << "Missing the mixed precision quantization config file "
                     << path;
      }
    }
  }
#endif
  return mixed_precision_quantization_config;
}

std::set<Node*>
ClearQuantInfoPass::GetTargetNodesFromMixedPrecisionQuantizationConfig(
    const std::unique_ptr<SSAGraph>& graph,
    const std::string& mixed_precision_quantization_config) {
  // Get target nodes from the mixed precision quantization config
  std::set<Node*> target_nodes;
  std::vector<std::string> lines =
      Split(mixed_precision_quantization_config, "\n");
  for (const auto& line : lines) {
    if (line.empty()) continue;
    std::vector<std::string> node_info = Split(line, ":");
    std::string op_type = node_info.at(0);
    std::vector<std::string> in_vars_name;
    if (node_info.size() > 1) {
      in_vars_name = Split(node_info.at(1), ",");
    }
    std::vector<std::string> out_vars_name;
    if (node_info.size() > 2) {
      out_vars_name = Split(node_info.at(2), ",");
    }

    for (auto& node : graph->mutable_nodes()) {
      if (node.IsArg()) continue;
      auto stmt = node.stmt();
      if (op_type != stmt->op_type()) continue;
      auto in_nodes = node.inlinks;
      auto out_nodes = node.outlinks;
      if (in_vars_name.size() > in_nodes.size() ||
          out_vars_name.size() > out_nodes.size()) {
        continue;
      }

      bool matched = true;

      for (auto in_var_name : in_vars_name) {
        bool find_var = false;
        for (auto* in_node : in_nodes) {
          if (in_node->arg()->name == in_var_name) {
            find_var = true;
            break;
          }
        }
        if (!find_var) {
          matched = false;
          break;
        }
      }

      for (auto out_var_name : out_vars_name) {
        bool find_var = false;
        for (auto* out_node : out_nodes) {
          if (out_node->arg()->name == out_var_name) {
            find_var = true;
            break;
          }
        }
        if (!find_var) {
          matched = false;
          break;
        }
      }

      if (matched) {
        target_nodes.insert(&node);
      }
    }
  }

  return target_nodes;
}

void ClearQuantInfo(paddle::lite::mir::Node* node) {
  if (node->IsArg()) return;
  auto op_desc = node->AsStmt().mutable_op_info();
  op_desc->DeleteAttr("bit_length");
  op_desc->DeleteAttr("enable_int8");

  std::vector<std::string> input_names;
  for (auto in_node : node->inlinks) {
    input_names.push_back(in_node->AsArg().name);
  }
  for (auto input_name : input_names) {
    std::string arg_name;
    int idx = -1;
    CHECK(op_desc->GetInputArgname(input_name, &arg_name));
    CHECK(op_desc->GetInputIndex(input_name, &idx));
    std::string scale_name = arg_name + std::to_string(idx) + "_scale";
    op_desc->DeleteAttr(scale_name);
  }

  std::vector<std::string> output_names;
  for (auto out_node : node->outlinks) {
    output_names.push_back(out_node->AsArg().name);
  }
  for (auto output_name : output_names) {
    std::string arg_name;
    int idx = -1;
    CHECK(op_desc->GetOutputArgname(output_name, &arg_name));
    CHECK(op_desc->GetOutputIndex(output_name, &idx));
    std::string scale_name = arg_name + std::to_string(idx) + "_scale";
    op_desc->DeleteAttr(scale_name);
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(clear_quant_info_pass, paddle::lite::mir::ClearQuantInfoPass)
    .BindTargets({TARGET(kNNAdapter)});
