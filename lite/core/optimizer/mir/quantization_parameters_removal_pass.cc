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
#include "lite/core/optimizer/mir/ssa_graph_utils.h"

namespace paddle {
namespace lite {
namespace mir {

void RestoreConv2DFromInt8ToFp32(Node* node) {
  auto op_desc = node->AsStmt().mutable_op_info();
  auto scope = node->AsStmt().op()->scope();
  const auto& op_type = op_desc->Type();
  auto filter_name = op_desc->Input("Filter").front();
  auto filter_tensor = scope->FindVar(filter_name)->GetMutable<lite::Tensor>();
  if (op_desc->HasInputScale(filter_name) &&
      filter_tensor->precision() == PRECISION(kInt8) &&
      filter_tensor->persistable()) {
    auto filter_scales = op_desc->GetInputScale(filter_name);
    auto filter_dims = filter_tensor->dims();
    Tensor temp_tensor;
    temp_tensor.CopyDataFrom(*filter_tensor);
    auto temp_data = temp_tensor.mutable_data<int8_t>();
    auto filter_data = filter_tensor->mutable_data<float>();
    auto groups = op_desc->GetAttr<int>("groups");
    auto output_channel_size = filter_dims[0];
    int64_t filter_outer_size = 1;
    auto filter_inner_size = filter_dims.production() / output_channel_size;
    if (op_type == "conv2d_transpose") {
      output_channel_size = filter_dims[1] * groups;
      filter_outer_size = filter_dims[0] / groups;
      filter_inner_size = filter_dims[2] * filter_dims[3];
    }
    CHECK_EQ(filter_scales.size(), output_channel_size);
    for (int64_t i = 0; i < filter_outer_size; i++) {
      for (int64_t j = 0; j < output_channel_size; j++) {
        for (int64_t k = 0; k < filter_inner_size; k++) {
          auto offset = i * output_channel_size * filter_inner_size +
                        j * filter_inner_size + k;
          filter_data[offset] = temp_data[offset] * filter_scales[j];
        }
      }
    }
    filter_tensor->set_precision(PRECISION(kFloat));
  }
}

void RestoreMatmulFromInt8ToFp32(Node* node) {
  auto op_desc = node->AsStmt().mutable_op_info();
  auto scope = node->AsStmt().op()->scope();
  const auto& op_type = op_desc->Type();
  auto weight_name = op_desc->Input("Y").front();
  auto weight_tensor = scope->FindVar(weight_name)->GetMutable<lite::Tensor>();
  if (op_desc->HasInputScale(weight_name) &&
      weight_tensor->precision() == PRECISION(kInt8) &&
      weight_tensor->persistable()) {
    auto weight_scales = op_desc->GetInputScale(weight_name);
    auto weight_dims = weight_tensor->dims();
    Tensor temp_tensor;
    temp_tensor.CopyDataFrom(*weight_tensor);
    auto temp_data = temp_tensor.mutable_data<int8_t>();
    auto weight_data = weight_tensor->mutable_data<float>();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK(op_type == "mul" ||
          (op_type == "matmul" && !op_desc->GetAttr<bool>("transpose_Y")) ||
          (op_type == "matmul_v2" && !op_desc->GetAttr<bool>("trans_y")));
    auto input_size = weight_dims[0];
    auto num_units = weight_dims[1];
    CHECK_EQ(weight_scales.size(), num_units);
    for (int64_t i = 0; i < input_size; i++) {
      for (int64_t j = 0; j < num_units; j++) {
        auto offset = i * num_units + j;
        weight_data[offset] = temp_data[offset] * weight_scales[j];
      }
    }
    weight_tensor->set_precision(PRECISION(kFloat));
  }
}

void QuantizationParametersRemovalPass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  std::string mixed_precision_quantization_configs;
  Scope* scope = nullptr;
  for (auto& any_op_node : graph->StmtTopologicalOrder()) {
    scope = any_op_node->AsStmt().op()->scope();
    if (scope) break;
  }
  CHECK(scope != nullptr);
#if defined(LITE_ON_MODEL_OPTIMIZE_TOOL) || defined(LITE_WITH_PYTHON) || \
    defined(LITE_WITH_NNADAPTER)
  // Load the mixed precision quantization configurations from APIs
  mixed_precision_quantization_configs = Context<TargetType::kNNAdapter>::
      NNAdapterMixedPrecisionQuantizationConfigBuffer(scope);
  if (mixed_precision_quantization_configs.empty()) {
    auto path = Context<TargetType::kNNAdapter>::
        NNAdapterMixedPrecisionQuantizationConfigPath(scope);
    if (!path.empty()) {
      std::vector<char> buffer;
      if (ReadFile(path, &buffer, false)) {
        if (!buffer.empty()) {
          mixed_precision_quantization_configs.insert(
              mixed_precision_quantization_configs.begin(),
              buffer.begin(),
              buffer.end());
        }
      } else {
        LOG(WARNING)
            << "Missing the mixed precision quantization configuration file "
            << path;
      }
    }
  }
#endif
  if (mixed_precision_quantization_configs.empty()) {
    mixed_precision_quantization_configs =
        GetConfigsFromEnv(MIXED_PRECISION_QUANTIZATION_CONFIG_FILE,
                          MIXED_PRECISION_QUANTIZATION_CONFIG_BUFFER);
  }
  if (mixed_precision_quantization_configs.empty()) return;
  auto op_nodes =
      GetNodesFromConfigs(graph.get(), mixed_precision_quantization_configs);
  for (auto& op_node : op_nodes) {
    CHECK(op_node->IsStmt());
    auto op_desc = op_node->AsStmt().mutable_op_info();
    const auto& op_type = op_desc->Type();
    // Restore the quantized weights of the target ops to fp32 precision
    if (op_type == "conv2d" || op_type == "depthwise_conv2d" ||
        op_type == "conv2d_transpose") {
      RestoreConv2DFromInt8ToFp32(op_node);
    } else if (op_type == "mul" || op_type == "matmul" ||
               op_type == "matmul_v2") {
      RestoreMatmulFromInt8ToFp32(op_node);
    }
    // Delete quantization-related attributes of the target ops
    op_desc->DeleteAttr("bit_length");
    op_desc->DeleteAttr("enable_int8");
    for (auto in_var_node : op_node->inlinks) {
      CHECK(in_var_node->IsArg());
      auto var_name = in_var_node->AsArg().name;
      std::string arg_name;
      int arg_idx = -1;
      CHECK(op_desc->GetInputArgname(var_name, &arg_name));
      CHECK(op_desc->GetInputIndex(var_name, &arg_idx));
      std::string scale_name = arg_name + std::to_string(arg_idx) + "_scale";
      op_desc->DeleteAttr(scale_name);
      auto& type = in_var_node->AsArg().type;
      if (type->precision() == PRECISION(kInt8)) {
        // TODO(zhupengyang): Only support trans to kFloat now. Other precision
        // should be considered.
        type = LiteType::GetTensorTy(
            type->target(), PRECISION(kFloat), type->layout());
      }
    }
    for (auto out_var_node : op_node->outlinks) {
      CHECK(out_var_node->IsArg());
      auto var_name = out_var_node->AsArg().name;
      std::string arg_name;
      int arg_idx = -1;
      CHECK(op_desc->GetOutputArgname(var_name, &arg_name));
      CHECK(op_desc->GetOutputIndex(var_name, &arg_idx));
      std::string scale_name = arg_name + std::to_string(arg_idx) + "_scale";
      op_desc->DeleteAttr(scale_name);
      auto& type = out_var_node->AsArg().type;
      if (type->precision() == PRECISION(kInt8)) {
        // TODO(zhupengyang): Only support trans to kFloat now. Other precision
        // should be considered.
        type = LiteType::GetTensorTy(
            type->target(), PRECISION(kFloat), type->layout());
      }
    }
    VLOG(5) << op_type << "'s quant params had been removed!";
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(quantization_parameters_removal_pass,
                  paddle::lite::mir::QuantizationParametersRemovalPass)
    .BindTargets({TARGET(kNNAdapter)});
