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

#include "lite/core/optimizer/mir/quantization_parameters_removal_pass.h"
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void QuantizationParametersRemovalPass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
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
  VLOG(3) << "find " << target_nodes.size() << " node matched.";
  for (auto node : target_nodes) {
    CHECK(node->IsStmt());
    ClearQuantInfo(node);
    for (auto out_node : node->outlinks) {
      auto& out_type = out_node->AsArg().type;
      if (out_type->precision() == PRECISION(kInt8)) {
        // TODO(zhupengyang): Only support trans to kFloat now. Other precision
        // should be considered.
        out_type = LiteType::GetTensorTy(
            out_type->target(), PRECISION(kFloat), out_type->layout());
      }
    }
  }
}

std::string
QuantizationParametersRemovalPass::GetMixedPrecisionQuantizationConfig(
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
          VLOG(3) << "Load mixed precision quantization config from file:\n"
                  << mixed_precision_quantization_config;
        }
      } else {
        LOG(WARNING) << "Missing the mixed precision quantization config file: "
                     << path;
      }
    }
  }
#endif
  return mixed_precision_quantization_config;
}

std::set<Node*> QuantizationParametersRemovalPass::
    GetTargetNodesFromMixedPrecisionQuantizationConfig(
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

void QuantizationParametersRemovalPass::ClearQuantInfo(
    paddle::lite::mir::Node* node) {
  if (node->IsArg()) return;
  auto op_desc = node->AsStmt().mutable_op_info();
  auto scope = node->AsStmt().op()->scope();
  const auto& op_type = op_desc->Type();
  VLOG(5) << "remove " << op_desc->Type() << " quant info.";

  if (op_type == "conv2d") {
    auto filter_name = op_desc->Input("Filter").front();
    auto filter_tensor =
        scope->FindVar(filter_name)->GetMutable<lite::Tensor>();
    if (op_desc->HasInputScale(filter_name) &&
        filter_tensor->precision() == PRECISION(kInt8)) {
      auto filter_scales = op_desc->GetInputScale(filter_name);
      auto filter_dims = filter_tensor->dims();
      Tensor temp_tensor;
      temp_tensor.CopyDataFrom(*filter_tensor);
      int8_t* temp_data = temp_tensor.mutable_data<int8_t>();
      float* filter_data = filter_tensor->mutable_data<float>();
      auto output_channel_size = filter_dims[0];
      CHECK_EQ(filter_scales.size(), output_channel_size);
      auto filter_inner_size = filter_dims.production() / output_channel_size;
      for (size_t i = 0; i < output_channel_size; i++) {
        for (size_t j = 0; j < filter_inner_size; j++) {
          filter_data[i * filter_inner_size + j] =
              temp_data[i * filter_inner_size + j] * filter_scales[i];
        }
      }
      filter_tensor->set_persistable(true);
      filter_tensor->set_precision(PRECISION(kFloat));
    }
  }

  op_desc->DeleteAttr("bit_length");
  op_desc->DeleteAttr("enable_int8");

  for (auto in_node : node->inlinks) {
    auto input_name = in_node->AsArg().name;
    std::string arg_name;
    int idx = -1;
    CHECK(op_desc->GetInputArgname(input_name, &arg_name));
    CHECK(op_desc->GetInputIndex(input_name, &idx));
    std::string scale_name = arg_name + std::to_string(idx) + "_scale";
    op_desc->DeleteAttr(scale_name);
  }

  for (auto out_node : node->outlinks) {
    auto output_name = out_node->AsArg().name;
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

REGISTER_MIR_PASS(quantization_parameters_removal_pass,
                  paddle::lite::mir::QuantizationParametersRemovalPass)
    .BindTargets({TARGET(kNNAdapter)});
