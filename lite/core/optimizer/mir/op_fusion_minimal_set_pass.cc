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

#include "lite/core/optimizer/mir/op_fusion_minimal_set_pass.h"
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {

#define SKIP_DELETE_INTERMEDIATE_NODES \
  for (auto& item : key2nodes_) {      \
    if (&item == &matched) {           \
      item.clear();                    \
    }                                  \
  }

class MulElementwiseAddFuser : public FuseBase {
 public:
  void BuildPattern() override {
    // Create the pattern nodes.
    auto mul_node = OpNode("mul", "mul")->AsIntermediate();
    auto mul_x_node = VarNode("mul_x")->assert_is_op_input("mul", "X");
    auto mul_y_node = VarNode("mul_y")
                          ->assert_is_op_input("mul", "Y")
                          ->assert_is_persistable_var();
    auto mul_out_node = VarNode("mul_out")
                            ->assert_is_op_output("mul", "Out")
                            ->assert_is_op_input("elementwise_add", "X")
                            ->AsIntermediate();
    auto elementwise_add_node =
        OpNode("elementwise_add", "elementwise_add")
            ->assert_node_satisfied([](const Node* node) -> bool {
              auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
              auto y_name = op_desc.Input("Y").front();
              auto scope = const_cast<Node*>(node)->AsStmt().op()->scope();
              auto y_tensor = scope->FindVar(y_name)->Get<lite::Tensor>();
              if (!y_tensor.persistable()) return false;
              return y_tensor.dims().size() == 1;
            })
            ->AsIntermediate();
    auto elementwise_add_y_node =
        VarNode("elementwise_add_y")
            ->assert_is_op_input("elementwise_add", "Y")
            ->assert_is_persistable_var();
    auto elementwise_add_out_node =
        VarNode("elementwise_add_out")
            ->assert_is_op_output("elementwise_add", "Out");
    // Create the topological connections for the above pattern nodes.
    std::vector<PMNode*> mul_inputs{mul_x_node, mul_y_node};
    std::vector<PMNode*> elementwise_add_inputs{mul_out_node,
                                                elementwise_add_y_node};
    mul_inputs >> *mul_node >> *mul_out_node;
    elementwise_add_inputs >> *elementwise_add_node >>
        *elementwise_add_out_node;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto mul_node = matched.at("mul");
    auto mul_op = mul_node->stmt()->op();
    auto scope = mul_op->scope();
    auto mul_x_node = matched.at("mul_x");
    auto mul_x_name = mul_x_node->arg()->name;
    auto mul_y_node = matched.at("mul_y");
    auto mul_y_name = mul_y_node->arg()->name;
    auto mul_y_var = scope->FindVar(mul_y_name);
    auto mul_y_dims = mul_y_var->Get<lite::Tensor>().dims();
    auto mul_out_node = matched.at("mul_out");
    auto mul_out_name = mul_out_node->arg()->name;
    auto elementwise_add_node = matched.at("elementwise_add");
    auto elementwise_add_y_node = matched.at("elementwise_add_y");
    auto elementwise_add_y_name = elementwise_add_y_node->arg()->name;
    auto elementwise_add_y_var = scope->FindVar(elementwise_add_y_name);
    auto elementwise_add_y_dims =
        elementwise_add_y_var->Get<lite::Tensor>().dims();
    auto elementwise_add_out_node = matched.at("elementwise_add_out");
    auto elementwise_add_out_name = elementwise_add_out_node->arg()->name;
    auto fc_num_units = mul_y_dims[1];
    if (elementwise_add_y_dims[0] != fc_num_units) {
      SKIP_DELETE_INTERMEDIATE_NODES
      LOG(WARNING) << "Op fusion failed! The dimension of the input Y of "
                      "elementwise_add should be ["
                   << fc_num_units << "], but recieve ["
                   << elementwise_add_y_dims[0] << "]!";
      return;
    }
    // Get the attributes from mul op and elementwise_add op.
    auto fc_desc = *mul_node->stmt()->op_info();
    auto elementwise_add_desc = *elementwise_add_node->stmt()->op_info();
    // Get input scales from mul op.
    std::vector<float> mul_x_scales;
    std::vector<float> mul_y_scales;
    if (fc_desc.HasInputScale(mul_x_name)) {
      mul_x_scales = fc_desc.GetInputScale(mul_x_name);
    }
    if (fc_desc.HasInputScale(mul_y_name)) {
      mul_y_scales = fc_desc.GetInputScale(mul_y_name);
    }
    // Modify the mul op desc for a new fc op.
    fc_desc.mutable_inputs()->clear();
    fc_desc.mutable_outputs()->clear();
    fc_desc.SetType("fc");
    fc_desc.SetInput("Input", {mul_x_name});
    fc_desc.SetInput("W", {mul_y_name});
    fc_desc.SetInput("Bias", {elementwise_add_y_name});
    fc_desc.SetOutput("Out", {elementwise_add_out_name});
    fc_desc.SetAttr("in_num_col_dims", fc_desc.GetAttr<int>("x_num_col_dims"));
    if (!mul_x_scales.empty()) {
      fc_desc.SetInputScale(mul_x_name, mul_x_scales);
    }
    if (!mul_y_scales.empty()) {
      fc_desc.SetInputScale(mul_y_name, mul_y_scales);
    }
    // Get the output threshold from elementwise_add op.
    if (elementwise_add_desc.HasAttr("out_threshold")) {
      fc_desc.SetAttr("out_threshold",
                      elementwise_add_desc.GetAttr<float>("out_threshold"));
    }
    // Compatible with a certain version of PaddleSlim, so @wanghaoshuang needs
    // to unify the name of the output threshold.
    if (elementwise_add_desc.HasAttr("Out0_threshold")) {
      fc_desc.SetAttr("Out0_threshold",
                      elementwise_add_desc.GetAttr<float>("Out0_threshold"));
    }
    // Create a new fc op with the op desc, and replace the matched subgraph
    // nodes.
    auto fc_op = LiteOpRegistry::Global().Create("fc");
    auto& valid_places = mul_op->valid_places();
    fc_op->Attach(fc_desc, scope);
    auto fc_node = graph->GraphCreateInstructNode(fc_op, valid_places);
    IR_NODE_LINK_TO(mul_x_node, fc_node);
    IR_NODE_LINK_TO(mul_y_node, fc_node);
    IR_NODE_LINK_TO(elementwise_add_y_node, fc_node);
    IR_OP_VAR_LINK(fc_node, elementwise_add_out_node);
  }
};

class Conv2dBatchNormFuser : public FuseBase {
 public:
  explicit Conv2dBatchNormFuser(const std::string& conv2d_type,
                                bool conv2d_bias,
                                const std::string& batch_norm_type)
      : conv2d_type_(conv2d_type),
        conv2d_bias_(conv2d_bias),
        batch_norm_type_(batch_norm_type) {}

  void BuildPattern() override {
    // Create the pattern nodes.
    auto conv2d_node = OpNode("conv2d", conv2d_type_);
    auto conv2d_input_node =
        VarNode("conv2d_input")->assert_is_op_input(conv2d_type_, "Input");
    auto conv2d_filter_node = VarNode("conv2d_filter")
                                  ->assert_is_op_input(conv2d_type_, "Filter")
                                  ->assert_is_persistable_var();
    auto conv2d_output_node = VarNode("conv2d_output")
                                  ->assert_is_op_output(conv2d_type_, "Output")
                                  ->assert_is_op_input(batch_norm_type_, "X")
                                  ->AsIntermediate();
    auto batch_norm_node =
        OpNode("batch_norm", batch_norm_type_)->AsIntermediate();
    auto batch_norm_scale_node =
        VarNode("batch_norm_scale")
            ->assert_is_op_input(batch_norm_type_, "Scale")
            ->assert_is_persistable_var()
            ->AsIntermediate();
    auto batch_norm_bias_node =
        VarNode("batch_norm_bias")
            ->assert_is_op_input(batch_norm_type_, "Bias")
            ->assert_is_persistable_var();
    auto batch_norm_mean_node =
        VarNode("batch_norm_mean")
            ->assert_is_op_input(batch_norm_type_, "Mean")
            ->assert_is_persistable_var()
            ->AsIntermediate();
    auto batch_norm_variance_node =
        VarNode("batch_norm_variance")
            ->assert_is_op_input(batch_norm_type_, "Variance")
            ->assert_is_persistable_var()
            ->AsIntermediate();
    auto batch_norm_y_node =
        VarNode("batch_norm_y")->assert_is_op_output(batch_norm_type_, "Y");
    auto batch_norm_mean_out_node =
        VarNode("batch_norm_mean_out")
            ->assert_is_op_output(batch_norm_type_, "MeanOut")
            ->AsIntermediate();
    auto batch_norm_variance_out_node =
        VarNode("batch_norm_variance_out")
            ->assert_is_op_output(batch_norm_type_, "VarianceOut")
            ->AsIntermediate();
    auto batch_norm_saved_mean_node =
        VarNode("batch_norm_saved_mean")
            ->assert_is_op_output(batch_norm_type_, "SavedMean")
            ->AsIntermediate();
    auto batch_norm_saved_variance_node =
        VarNode("batch_norm_saved_variance")
            ->assert_is_op_output(batch_norm_type_, "SavedVariance")
            ->AsIntermediate();
    // Create the topological connections for the above pattern nodes.
    std::vector<PMNode*> conv2d_inputs{conv2d_input_node, conv2d_filter_node};
    if (conv2d_bias_) {
      auto conv2d_bias_node = VarNode("conv2d_bias")
                                  ->assert_is_op_input(conv2d_type_, "Bias")
                                  ->assert_is_persistable_var()
                                  ->AsIntermediate();
      conv2d_inputs.emplace_back(conv2d_bias_node);
    }
    std::vector<PMNode*> batch_norm_inputs{conv2d_output_node,
                                           batch_norm_scale_node,
                                           batch_norm_bias_node,
                                           batch_norm_mean_node,
                                           batch_norm_variance_node};
    std::vector<PMNode*> batch_norm_outputs{batch_norm_y_node,
                                            batch_norm_mean_out_node,
                                            batch_norm_variance_out_node,
                                            batch_norm_saved_mean_node,
                                            batch_norm_saved_variance_node};
    conv2d_inputs >> *conv2d_node >> *conv2d_output_node;
    batch_norm_inputs >> *batch_norm_node >> batch_norm_outputs;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto conv2d_node = matched.at("conv2d");
    auto conv2d_op = conv2d_node->stmt()->op();
    auto scope = conv2d_op->scope();
    auto conv2d_filter_node = matched.at("conv2d_filter");
    auto conv2d_filter_name = conv2d_filter_node->arg()->name;
    auto conv2d_filter_var = scope->FindVar(conv2d_filter_name);
    auto conv2d_filter_tensor = conv2d_filter_var->GetMutable<lite::Tensor>();
    auto conv2d_filter_dims = conv2d_filter_tensor->dims();
    auto batch_norm_node = matched.at("batch_norm");
    auto batch_norm_node_op = batch_norm_node->stmt()->op();
    auto batch_norm_scale_node = matched.at("batch_norm_scale");
    auto batch_norm_scale_name = batch_norm_scale_node->arg()->name;
    auto batch_norm_scale_var = scope->FindVar(batch_norm_scale_name);
    auto batch_norm_scale_tensor = batch_norm_scale_var->Get<lite::Tensor>();
    auto batch_norm_scale_dims = batch_norm_scale_tensor.dims();
    auto batch_norm_bias_node = matched.at("batch_norm_bias");
    auto batch_norm_bias_name = batch_norm_bias_node->arg()->name;
    auto batch_norm_bias_var = scope->FindVar(batch_norm_bias_name);
    auto batch_norm_bias_tensor =
        batch_norm_bias_var->GetMutable<lite::Tensor>();
    auto batch_norm_bias_dims = batch_norm_bias_tensor->dims();
    auto batch_norm_mean_node = matched.at("batch_norm_mean");
    auto batch_norm_mean_name = batch_norm_mean_node->arg()->name;
    auto batch_norm_mean_var = scope->FindVar(batch_norm_mean_name);
    auto batch_norm_mean_tensor = batch_norm_mean_var->Get<lite::Tensor>();
    auto batch_norm_mean_dims = batch_norm_mean_tensor.dims();
    auto batch_norm_variance_node = matched.at("batch_norm_variance");
    auto batch_norm_variance_name = batch_norm_variance_node->arg()->name;
    auto batch_norm_variance_var = scope->FindVar(batch_norm_variance_name);
    auto batch_norm_variance_tensor =
        batch_norm_variance_var->Get<lite::Tensor>();
    auto batch_norm_variance_dims = batch_norm_variance_tensor.dims();
    auto batch_norm_y_node = matched.at("batch_norm_y");
    auto batch_norm_y_name = batch_norm_y_node->arg()->name;
    // Get the attributes from conv2d op and batch_norm op.
    auto conv2d_desc = *conv2d_node->stmt()->op_info();
    auto batch_norm_desc = *batch_norm_node->stmt()->op_info();
    auto conv2d_groups = conv2d_desc.GetAttr<int>("groups");
    auto batch_norm_eps = batch_norm_desc.GetAttr<float>("epsilon");
    auto conv2d_output_channel_size = conv2d_filter_dims[0];
    if (conv2d_type_ == "conv2d_transpose") {
      conv2d_output_channel_size = conv2d_filter_dims[1] * conv2d_groups;
    }
    if (batch_norm_scale_dims[0] != conv2d_output_channel_size ||
        batch_norm_bias_dims[0] != conv2d_output_channel_size ||
        batch_norm_mean_dims[0] != conv2d_output_channel_size ||
        batch_norm_variance_dims[0] != conv2d_output_channel_size) {
      SKIP_DELETE_INTERMEDIATE_NODES
      LOG(WARNING) << "Op fusion failed! The dimension of the input Scale, "
                      "Bias, Mean and Variance of "
                   << batch_norm_type_ << " should be ["
                   << conv2d_output_channel_size << "], but recieve ["
                   << batch_norm_scale_dims[0] << "], ["
                   << batch_norm_bias_dims[0] << "], ["
                   << batch_norm_mean_dims[0] << "] and ["
                   << batch_norm_variance_dims[0] << "]!";
      return;
    }
    // Compute the alpha and beta of batch_norm op as the following formula:
    // alpha[channel_idx] = scale[channel_idx] / (sqrt(variance[channel_idx]) +
    // eps)
    // beta[channel_idx] = (-mean[channel_idx]) * alpha[channel_idx]
    Tensor batch_norm_alpha_tensor, batch_norm_beta_tensor;
    batch_norm_alpha_tensor.Resize(
        std::vector<int64_t>({conv2d_output_channel_size}));
    batch_norm_beta_tensor.Resize(
        std::vector<int64_t>({conv2d_output_channel_size}));
    auto batch_norm_alpha_data = batch_norm_alpha_tensor.mutable_data<float>();
    auto batch_norm_beta_data = batch_norm_beta_tensor.mutable_data<float>();
    auto batch_norm_scale_data = batch_norm_scale_tensor.data<float>();
    auto batch_norm_mean_data = batch_norm_mean_tensor.data<float>();
    auto batch_norm_variance_data = batch_norm_variance_tensor.data<float>();
    for (int64_t i = 0; i < conv2d_output_channel_size; i++) {
      batch_norm_alpha_data[i] =
          batch_norm_scale_data[i] /
          std::sqrt(batch_norm_variance_data[i] + batch_norm_eps);
      batch_norm_beta_data[i] =
          (-batch_norm_mean_data[i]) * batch_norm_alpha_data[i];
    }
    std::vector<float> conv2d_filter_scales;
    if (conv2d_desc.HasInputScale(conv2d_filter_name)) {
      conv2d_filter_scales = conv2d_desc.GetInputScale(conv2d_filter_name);
      auto conv2d_filter_data = conv2d_filter_tensor->mutable_data<int8_t>();
      if (conv2d_type_ == "conv2d_transpose") {
      } else {
        auto conv2d_filter_inner_size =
            conv2d_filter_dims.production() / conv2d_output_channel_size;
        for (int64_t i = 0; i < conv2d_output_channel_size; i++) {
          conv2d_filter_scales[i] *= fabsf(batch_norm_alpha_data[i]);
          if (batch_norm_alpha_data[i] >= 0.f) continue;
          for (int64_t j = 0; j < conv2d_filter_inner_size; j++) {
            conv2d_filter_data[i * conv2d_filter_inner_size + j] *= -1;
          }
        }
      }
    } else {
      auto conv2d_filter_data = conv2d_filter_tensor->mutable_data<float>();
      if (conv2d_type_ == "conv2d_transpose") {
      } else {
        int64_t conv2d_filter_inner_size =
            conv2d_filter_dims.production() / conv2d_output_channel_size;
        for (int64_t i = 0; i < conv2d_output_channel_size; i++) {
          for (int64_t j = 0; j < conv2d_filter_inner_size; j++) {
            conv2d_filter_data[i * conv2d_filter_inner_size + j] *=
                batch_norm_alpha_data[i];
          }
        }
      }
    }
    // Merge bias values if bias already exists in conv2d
    auto batch_norm_bias_data = batch_norm_bias_tensor->mutable_data<float>();
    if (conv2d_bias_) {
      auto conv2d_bias_node = matched.at("conv2d_bias");
      auto conv2d_bias_name = conv2d_bias_node->arg()->name;
      auto conv2d_bias_var = scope->FindVar(conv2d_bias_name);
      auto conv2d_bias_tensor = conv2d_bias_var->Get<lite::Tensor>();
      auto conv2d_bias_dims = conv2d_bias_tensor.dims();
      CHECK_EQ(conv2d_bias_dims.size(), 1);
      CHECK_EQ(conv2d_bias_dims[0], conv2d_output_channel_size);
      auto conv2d_bias_data = conv2d_bias_tensor.data<float>();
      for (int64_t i = 0; i < conv2d_output_channel_size; i++) {
        batch_norm_bias_data[i] +=
            batch_norm_alpha_data[i] * conv2d_bias_data[i];
      }
    }
    for (int64_t i = 0; i < conv2d_output_channel_size; i++) {
      batch_norm_bias_data[i] += batch_norm_beta_data[i];
    }
    // Update the conv2d op desc and links
    conv2d_desc.SetInput("Bias", {batch_norm_bias_name});
    conv2d_desc.SetOutput("Output", {batch_norm_y_name});
    if (!conv2d_filter_scales.empty()) {
      conv2d_desc.SetInputScale(conv2d_filter_name, conv2d_filter_scales);
    }
    // Set the output threshold from batch_norm op.
    if (batch_norm_desc.HasAttr("out_threshold")) {
      conv2d_desc.SetAttr("out_threshold",
                          batch_norm_desc.GetAttr<float>("out_threshold"));
    }
    // Compatible with a certain version of PaddleSlim, so @wanghaoshuang needs
    // to unify the name of the output threshold.
    if (batch_norm_desc.HasAttr("Y0_threshold")) {
      conv2d_desc.SetAttr("Output0_threshold",
                          batch_norm_desc.GetAttr<float>("Y0_threshold"));
    }
    auto& valid_places = conv2d_op->valid_places();
    conv2d_node->stmt()->ResetOp(conv2d_desc, valid_places);
    IR_NODE_LINK_TO(batch_norm_bias_node, conv2d_node);
    IR_OP_VAR_LINK(conv2d_node, batch_norm_y_node);
  }

 private:
  std::string conv2d_type_{"conv2d"};
  bool conv2d_bias_{false};
  std::string batch_norm_type_{"batch_norm"};
};

class Conv2dElementwiseAddFuser : public FuseBase {
 public:
  explicit Conv2dElementwiseAddFuser(const std::string& conv2d_type,
                                     bool conv2d_bias)
      : conv2d_type_(conv2d_type), conv2d_bias_(conv2d_bias) {}

  void BuildPattern() override {
    // Create the pattern nodes.
    auto conv2d_node = OpNode("conv2d", conv2d_type_);
    auto conv2d_input_node =
        VarNode("conv2d_input")->assert_is_op_input(conv2d_type_, "Input");
    auto conv2d_filter_node = VarNode("conv2d_filter")
                                  ->assert_is_op_input(conv2d_type_, "Filter")
                                  ->assert_is_persistable_var();
    auto conv2d_output_node = VarNode("conv2d_output")
                                  ->assert_is_op_output(conv2d_type_, "Output")
                                  ->assert_is_op_input("elementwise_add", "X")
                                  ->AsIntermediate();
    auto elementwise_add_node =
        OpNode("elementwise_add", "elementwise_add")
            ->assert_node_satisfied([](const Node* node) -> bool {
              auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
              auto y_name = op_desc.Input("Y").front();
              auto scope = const_cast<Node*>(node)->AsStmt().op()->scope();
              auto y_tensor = scope->FindVar(y_name)->Get<lite::Tensor>();
              if (!y_tensor.persistable()) return false;
              return y_tensor.dims().size() == 1;
            })
            ->AsIntermediate();
    auto elementwise_add_y_node =
        VarNode("elementwise_add_y")
            ->assert_is_op_input("elementwise_add", "Y")
            ->assert_is_persistable_var();
    auto elementwise_add_out_node =
        VarNode("elementwise_add_out")
            ->assert_is_op_output("elementwise_add", "Out");
    // Create the topological connections for the above pattern nodes.
    std::vector<PMNode*> conv2d_inputs{conv2d_input_node, conv2d_filter_node};
    if (conv2d_bias_) {
      auto conv2d_bias_node = VarNode("conv2d_bias")
                                  ->assert_is_op_input(conv2d_type_, "Bias")
                                  ->assert_is_persistable_var()
                                  ->AsIntermediate();
      conv2d_inputs.emplace_back(conv2d_bias_node);
    }
    std::vector<PMNode*> elementwise_add_inputs{conv2d_output_node,
                                                elementwise_add_y_node};
    conv2d_inputs >> *conv2d_node >> *conv2d_output_node;
    elementwise_add_inputs >> *elementwise_add_node >>
        *elementwise_add_out_node;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto conv2d_node = matched.at("conv2d");
    auto conv2d_op = conv2d_node->stmt()->op();
    auto scope = conv2d_op->scope();
    auto conv2d_filter_node = matched.at("conv2d_filter");
    auto conv2d_filter_name = conv2d_filter_node->arg()->name;
    auto conv2d_filter_var = scope->FindVar(conv2d_filter_name);
    auto conv2d_filter_dims = conv2d_filter_var->Get<lite::Tensor>().dims();
    auto elementwise_add_node = matched.at("elementwise_add");
    auto elementwise_add_op = elementwise_add_node->stmt()->op();
    auto elementwise_add_y_node = matched.at("elementwise_add_y");
    auto elementwise_add_y_name = elementwise_add_y_node->arg()->name;
    auto elementwise_add_y_var = scope->FindVar(elementwise_add_y_name);
    auto elementwise_add_y_tensor =
        elementwise_add_y_var->GetMutable<lite::Tensor>();
    auto elementwise_add_y_dims = elementwise_add_y_tensor->dims();
    auto elementwise_add_out_node = matched.at("elementwise_add_out");
    auto elementwise_add_out_name = elementwise_add_out_node->arg()->name;
    // Get the attributes from conv2d op and elementwise_add op.
    auto conv2d_desc = *conv2d_node->stmt()->op_info();
    auto elementwise_add_desc = *elementwise_add_node->stmt()->op_info();
    auto conv2d_groups = conv2d_desc.GetAttr<int>("groups");
    auto conv2d_output_channel_size = conv2d_filter_dims[0];
    if (conv2d_type_ == "conv2d_transpose") {
      conv2d_output_channel_size = conv2d_filter_dims[1] * conv2d_groups;
    }
    if (elementwise_add_y_dims[0] != conv2d_output_channel_size) {
      SKIP_DELETE_INTERMEDIATE_NODES
      LOG(WARNING) << "Op fusion failed! The dimension of the input Y of "
                      "elementwise_add should be ["
                   << conv2d_output_channel_size << "], but recieve ["
                   << elementwise_add_y_dims[0] << "]!";
      return;
    }
    // Merge bias values if bias already exists in conv2d
    if (conv2d_bias_) {
      auto conv2d_bias_node = matched.at("conv2d_bias");
      auto conv2d_bias_name = conv2d_bias_node->arg()->name;
      auto conv2d_bias_var = scope->FindVar(conv2d_bias_name);
      auto conv2d_bias_tensor = conv2d_bias_var->Get<lite::Tensor>();
      auto conv2d_bias_dims = conv2d_bias_tensor.dims();
      CHECK_EQ(conv2d_bias_dims.size(), 1);
      CHECK_EQ(conv2d_bias_dims[0], conv2d_output_channel_size);
      auto conv2d_bias_data = conv2d_bias_tensor.data<float>();
      auto elementwise_add_y_data =
          elementwise_add_y_tensor->mutable_data<float>();
      for (int64_t i = 0; i < conv2d_output_channel_size; i++) {
        elementwise_add_y_data[i] += conv2d_bias_data[i];
      }
    }
    // Update the conv2d op desc and links
    conv2d_desc.SetInput("Bias", {elementwise_add_y_name});
    conv2d_desc.SetOutput("Output", {elementwise_add_out_name});
    // Get the output threshold from elementwise_add op.
    if (elementwise_add_desc.HasAttr("out_threshold")) {
      conv2d_desc.SetAttr("out_threshold",
                          elementwise_add_desc.GetAttr<float>("out_threshold"));
    }
    // Compatible with a certain version of PaddleSlim, so @wanghaoshuang needs
    // to unify the name of the output threshold.
    if (elementwise_add_desc.HasAttr("Out0_threshold")) {
      conv2d_desc.SetAttr(
          "Output0_threshold",
          elementwise_add_desc.GetAttr<float>("Out0_threshold"));
    }
    auto& valid_places = conv2d_op->valid_places();
    conv2d_node->stmt()->ResetOp(conv2d_desc, valid_places);
    IR_NODE_LINK_TO(elementwise_add_y_node, conv2d_node);
    IR_OP_VAR_LINK(conv2d_node, elementwise_add_out_node);
  }

 private:
  std::string conv2d_type_{"conv2d"};
  bool conv2d_bias_{false};
};

class Conv2dActivationFuser : public FuseBase {
 public:
  explicit Conv2dActivationFuser(const std::string& conv2d_type,
                                 bool conv2d_bias,
                                 const std::string& act_type)
      : conv2d_type_(conv2d_type),
        conv2d_bias_(conv2d_bias),
        act_type_(act_type) {}

  void BuildPattern() override {
    // Create the pattern nodes.
    auto conv2d_node = OpNode("conv2d", conv2d_type_);
    auto conv2d_input_node =
        VarNode("conv2d_input")->assert_is_op_input(conv2d_type_, "Input");
    auto conv2d_filter_node = VarNode("conv2d_filter")
                                  ->assert_is_op_input(conv2d_type_, "Filter")
                                  ->assert_is_persistable_var();
    auto conv2d_output_node = VarNode("conv2d_output")
                                  ->assert_is_op_output(conv2d_type_, "Output")
                                  ->assert_is_op_input(act_type_, "X")
                                  ->AsIntermediate();
    auto act_node = OpNode("act", act_type_)->AsIntermediate();
    auto act_out_node =
        VarNode("act_out")->assert_is_op_output(act_type_, "Out");
    // Create the topological connections for the above pattern nodes.
    std::vector<PMNode*> conv2d_inputs{conv2d_input_node, conv2d_filter_node};
    if (conv2d_bias_) {
      auto conv2d_bias_node = VarNode("conv2d_bias")
                                  ->assert_is_op_input(conv2d_type_, "Bias")
                                  ->assert_is_persistable_var();
      conv2d_inputs.emplace_back(conv2d_bias_node);
    }
    conv2d_inputs >> *conv2d_node >> *conv2d_output_node;
    *conv2d_output_node >> *act_node >> *act_out_node;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto conv2d_node = matched.at("conv2d");
    auto conv2d_op = conv2d_node->stmt()->op();
    auto act_node = matched.at("act");
    auto act_out_node = matched.at("act_out");
    auto act_out_name = act_out_node->arg()->name;
    // Get the attributes from conv2d op and act op.
    auto conv2d_desc = *conv2d_node->stmt()->op_info();
    auto act_desc = *act_node->stmt()->op_info();
    // Update the conv2d op desc and links
    conv2d_desc.SetAttr("with_act", true);
    conv2d_desc.SetAttr("act_type", act_type_);
    if (act_type_ == "relu") {
      conv2d_desc.SetAttr("fuse_relu", true);
    }
    if (act_type_ == "relu6") {
      float alpha = act_desc.GetAttr<float>("threshold");
      conv2d_desc.SetAttr("fuse_brelu_threshold", alpha);
    }
    conv2d_desc.SetOutput("Output", {act_out_name});
    // Get the output threshold from act op.
    if (act_desc.HasAttr("out_threshold")) {
      conv2d_desc.SetAttr("out_threshold",
                          act_desc.GetAttr<float>("out_threshold"));
    }
    // Compatible with a certain version of PaddleSlim, so @wanghaoshuang needs
    // to unify the name of the output threshold.
    if (act_desc.HasAttr("Out0_threshold")) {
      conv2d_desc.SetAttr("Output0_threshold",
                          act_desc.GetAttr<float>("Out0_threshold"));
    }
    auto& valid_places = conv2d_op->valid_places();
    conv2d_node->stmt()->ResetOp(conv2d_desc, valid_places);
    IR_OP_VAR_LINK(conv2d_node, act_out_node);
  }

 private:
  std::string conv2d_type_{"conv2d"};
  bool conv2d_bias_{false};
  std::string act_type_{"relu"};
};

void ApplyMulElementwiseAddFuser(SSAGraph* graph) {
  MulElementwiseAddFuser fuser;
  fuser(graph);
}

void ApplyMatmulElementwiseAddFuser(SSAGraph* graph) {
  // MatmulElementwiseAddFuser fuser;
  // fuser(graph);
}

void ApplyFCActivationFuser(SSAGraph* graph) {
  // FCActivationFuser fuser;
  // fuser(graph);
}

void ApplyConv2dBatchNormFuser(SSAGraph* graph) {
  for (auto conv2d_bias : {true, false}) {
    for (auto conv2d_type :
         {"conv2d", "depthwise_conv2d", "conv2d_transpose"}) {
      for (auto batch_norm_type : {"batch_norm", "sync_batch_norm"}) {
        VLOG(5) << "conv2d_type:" << conv2d_type
                << " conv2d_bias:" << conv2d_bias
                << " batch_norm_type:" << batch_norm_type;
        Conv2dBatchNormFuser fuser(conv2d_type, conv2d_bias, batch_norm_type);
        fuser(graph);
      }
    }
  }
}

void ApplyConv2dElementwiseAddFuser(SSAGraph* graph) {
  // The case of conv2d_bias=true should be handled first
  for (auto conv2d_bias : {true, false}) {
    for (auto conv2d_type :
         {"conv2d", "depthwise_conv2d", "conv2d_transpose"}) {
      VLOG(5) << "conv2d_type:" << conv2d_type
              << " conv2d_bias:" << conv2d_bias;
      Conv2dElementwiseAddFuser fuser(conv2d_type, conv2d_bias);
      fuser(graph);
    }
  }
}

void ApplyConv2dActivationFuser(SSAGraph* graph) {
  for (auto conv2d_bias : {true, false}) {
    for (auto conv2d_type :
         {"conv2d", "depthwise_conv2d", "conv2d_transpose"}) {
      for (auto act_type : {"relu", "relu1", "relu6"}) {
        VLOG(5) << "conv2d_type:" << conv2d_type
                << " conv2d_bias:" << conv2d_bias << " act_type:" << act_type;
        Conv2dActivationFuser fuser(conv2d_type, conv2d_bias, act_type);
        fuser(graph);
      }
    }
  }
}

void ApplyReshapeTransposeReshapeFuser(SSAGraph* graph) {
  // ReshapeTransposeReshapeFuser fuser;
  // fuser(graph);
}

void OpFusionMinimalSetPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  ApplyMulElementwiseAddFuser(graph.get());
  ApplyMatmulElementwiseAddFuser(graph.get());
  ApplyFCActivationFuser(graph.get());
  ApplyConv2dBatchNormFuser(graph.get());
  ApplyConv2dElementwiseAddFuser(graph.get());
  ApplyConv2dActivationFuser(graph.get());
  // Since some hardware does not support 5-D inputs and outputs, and the
  // shuffle channel op is more general and friendly to hardware manufacturers,
  // it is necessary to convert reshape+transpose+reshape to shuffle channel op.
  ApplyReshapeTransposeReshapeFuser(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(op_fusion_minimal_set_pass,
                  paddle::lite::mir::OpFusionMinimalSetPass)
    .BindTargets({TARGET(kNNAdapter)});
